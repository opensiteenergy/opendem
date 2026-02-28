import gc
import numpy as np
import os
import yaml
import signal
import math
import time
from osgeo import gdal, ogr, osr

# Standard GIS exception handling
gdal.UseExceptions()

class OpenDEM:
    def __init__(self, config_path):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_dir = self.config.get('cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Optimized GDAL settings
        gdal.SetConfigOption('GDAL_HTTP_CACHE', 'YES')
        gdal.SetConfigOption('GDAL_HTTP_CACHE_DIRECTORY', self.cache_dir)
        gdal.SetConfigOption('GDAL_CACHEMAX', '1024') 
        gdal.SetConfigOption('CPL_DEBUG', 'ON')
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'YES') 
        gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '10')
        gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '5')
        gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '120')

        self._last_gdal_p = -1

    def _handle_interrupt(self, sig, frame):
        self.log("Intercepted ctrl+C. Forcing exit...")
        os._exit(0)

    def log(self, message):
        print(f"[opendem] {message}")

    def _calculate_zoom_level(self, res):
        # res is meters per pixel
        # Standard formula: zoom = log2(circumference / (res * 256))
        # For Z=15, res is roughly 4.8m at equator, ~2.7m in UK
        import math
        
        # Calculate theoretical zoom
        z = math.ceil(math.log2(40075016.686 / (res * 256.0)))
        
        # CAP AT 15: Terrarium's reliable limit for the UK
        final_zoom = min(int(z), 15)
        
        self.log(f"Requested Res: {res}m -> Theoretical Zoom: {z} -> Capped Zoom: {final_zoom}")
        return final_zoom

    def _generate_vrt(self, internal_res):
        vrt_path = os.path.join(self.cache_dir, "source.vrt")
        zoom = self._calculate_zoom_level(internal_res)
        self.log(f"Selecting TileLevel {zoom} for {internal_res:.2f}m precision")
        
        vrt_content = f"""<GDAL_WMS>
    <Service name="TMS">
        <ServerUrl>{self.config['source']}</ServerUrl>
    </Service>
    <DataWindow>
        <UpperLeftX>-20037508.34</UpperLeftX>
        <UpperLeftY>20037508.34</UpperLeftY>
        <LowerRightX>20037508.34</LowerRightX>
        <LowerRightY>-20037508.34</LowerRightY>
        <TileLevel>{zoom}</TileLevel>
        <YOrigin>top</YOrigin>
    </DataWindow>
    <Projection>EPSG:3857</Projection>
    <BlockSizeX>256</BlockSizeX>
    <BlockSizeY>256</BlockSizeY>
    <BandsCount>3</BandsCount>
    <Cache>
        <Path>{os.path.abspath(self.cache_dir)}</Path>
        <CacheMaxAge>315360000</CacheMaxAge>
        <Unique>True</Unique>
    </Cache>
</GDAL_WMS>"""
        with open(vrt_path, "w") as f:
            f.write(vrt_content.strip())
        return vrt_path

    def _prepare_output_raster(self, path, bounds, res, dtype=gdal.GDT_Float32, nodata=-9999):
        srs_in = osr.SpatialReference(); srs_in.ImportFromEPSG(4326)
        srs_out = osr.SpatialReference(); srs_out.ImportFromEPSG(3857)
        transform = osr.CoordinateTransformation(srs_in, srs_out)
        
        minx, miny, _ = transform.TransformPoint(bounds[0], bounds[1])
        maxx, maxy, _ = transform.TransformPoint(bounds[2], bounds[3])

        width = int((maxx - minx) / res)
        height = int((maxy - miny) / res)

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(path, width, height, 1, dtype, options=['TILED=YES', 'COMPRESS=LZW'])
        ds.SetProjection(srs_out.ExportToWkt())
        ds.SetGeoTransform([minx, res, 0, maxy, 0, -res])
        ds.GetRasterBand(1).SetNoDataValue(nodata)
        return ds

    def _get_buffered_clipping_geom(self, target_res):
        path = self.config.get('clipping')
        if not path: 
            return None
        
        # Unique cache for this specific operation
        cache_path = os.path.join(self.cache_dir, "clip_processed_cache.gpkg")
        
        # 1. Check Cache First
        if os.path.exists(cache_path):
            self.log(f"Found {cache_path}. Loading instantly...")
            ds_c = ogr.Open(cache_path)
            if ds_c:
                lyr_c = ds_c.GetLayer()
                feat = lyr_c.GetNextFeature()
                if feat:
                    return feat.GetGeometryRef().Clone()

        # 2. Process Raw Data
        self.log(f"Cache missing. Processing {path}...")
        ds = ogr.Open(path)
        if ds is None: 
            raise FileNotFoundError(f"Could not open {path}")
            
        layer = ds.GetLayer()
        src_srs = layer.GetSpatialRef() or osr.SpatialReference()
        if not layer.GetSpatialRef(): 
            src_srs.ImportFromEPSG(4326)
        
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(3857)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tx = osr.CoordinateTransformation(src_srs, tgt_srs)

        # 3. Simplify-before-Union (The "Anti-Hang" Logic)
        self.log("Simplifying and transforming features individually...")
        simple_geoms = []
        for feat in layer:
            g = feat.GetGeometryRef().Clone()
            g.Transform(tx)
            
            # Simplify each feature by 100m. This clears out the "fractal" 
            # coastline detail that chokes the Union engine.
            g_simple = g.Simplify(100.0)
            simple_geoms.append(g_simple)

        self.log(f"Unioning {len(simple_geoms)} simplified features...")
        combined = ogr.Geometry(ogr.wkbMultiPolygon)
        
        # We use Union() on the list of low-poly shapes
        for g in simple_geoms:
            combined = combined.Union(g)

        # 4. Final Buffer (1km safety margin)
        self.log("Applying 1000m safety buffer...")
        final_geom = combined.Buffer(1000.0)
        
        # Clean up vertex count again after buffering if needed
        final_geom = final_geom.Simplify(50.0)

        # 5. Save result so we never do this again
        if os.path.exists(cache_path): 
            ogr.GetDriverByName("GPKG").DeleteDataSource(cache_path)
        
        out_ds = ogr.GetDriverByName("GPKG").CreateDataSource(cache_path)
        out_lyr = out_ds.CreateLayer("clip", tgt_srs, ogr.wkbMultiPolygon)
        f = ogr.Feature(out_lyr.GetLayerDefn())
        f.SetGeometry(final_geom)
        out_lyr.CreateFeature(f)
        out_ds.FlushCache()
        
        self.log("Clipping geometry prepared and cached.")
        return final_geom

    def run(self):
        start_time = time.time()
        res = float(self.config['resolution'])
        p_type = self.config.get('process', 'elevation')
        
        # Internal res for sampling (1m is very heavy, consider 5m or 10m for 1km output)
        i_res = min(res / 2.0, 1.0) if p_type == 'slope' else res
        i_res = 3
        vrt = self._generate_vrt(i_res)
        
        out_tmp = os.path.join(self.cache_dir, "temp_processed.tif")
        
        # Ensure 3857 SRS for internal processing
        srs_3857 = osr.SpatialReference(); srs_3857.ImportFromEPSG(3857)
        srs_4326 = osr.SpatialReference(); srs_4326.ImportFromEPSG(4326)
        srs_3857.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tx = osr.CoordinateTransformation(srs_4326, srs_3857)
        
        # Transform Bounds carefully
        b = self.config['bounds']
        p_min = tx.TransformPoint(b[0], b[1])
        p_max = tx.TransformPoint(b[2], b[3])
        
        # Extract correct Min/Max to handle flipped axes
        xmin, xmax = sorted([p_min[0], p_max[0]])
        ymin, ymax = sorted([p_min[1], p_max[1]])

        cols = math.ceil((xmax - xmin) / res)
        rows = math.ceil((ymax - ymin) / res)
        
        self.log(f"Grid Dimensions: {cols}x{rows} pixels at {res}m")

        driver = gdal.GetDriverByName("GTiff")
        ds_final = driver.Create(out_tmp, cols, rows, 1, gdal.GDT_Float32)
        # Note: Origin is Top-Left, so we use xmin and ymax (highest Y)
        ds_final.SetGeoTransform([xmin, res, 0, ymax, 0, -res])
        ds_final.SetProjection(srs_3857.ExportToWkt())
        ds_final.GetRasterBand(1).SetNoDataValue(-9999)

        clip_geom = self._get_buffered_clipping_geom(res)
        
        self._run_tiled_slope(vrt, ds_final, i_res, clip_geom)

        ds_final.FlushCache()
        ds_final = None 
        
        self._post_process(out_tmp, self.config['output'])
        self.log(f"Total processing time: {(time.time()-start_time)/60:.2f} minutes.")

    def _run_tiled_slope(self, vrt, ds_final, i_res, clip_geom):
        gt = ds_final.GetGeoTransform()
        x_sz, y_sz = ds_final.RasterXSize, ds_final.RasterYSize
        chunk = 256 # Smaller chunk for better progress visibility
        
        num_x = math.ceil(x_sz / chunk)
        num_y = math.ceil(y_sz / chunk)
        total = num_x * num_y
        count = 0

        self.log(f"Starting Tiled Processing ({total} chunks total)...")

        srs_3857 = osr.SpatialReference(); srs_3857.ImportFromEPSG(3857)
        srs_4326 = osr.SpatialReference(); srs_4326.ImportFromEPSG(4326)
        srs_3857.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct_to_4326 = osr.CoordinateTransformation(srs_3857, srs_4326)

        for y in range(0, y_sz, chunk):
            rows = min(chunk, y_sz - y)
            gc.collect()
            for x in range(0, x_sz, chunk):
                cols = min(chunk, x_sz - x)
                count += 1
                
                # Calculate the 3857 coordinates for this chunk
                # x_coord = origin_x + pixel_index * res
                # y_coord = origin_y + pixel_index * -res
                cx0 = gt[0] + x * gt[1]
                cy1 = gt[3] + y * gt[5]
                cx1 = cx0 + cols * gt[1]
                cy0 = cy1 + rows * gt[5]
                
                p_lon, p_lat, _ = ct_to_4326.TransformPoint(cx0, cy1)
                print(p_lon, p_lat)
                
                # Check Intersection
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(cx0, cy0)
                ring.AddPoint(cx1, cy0)
                ring.AddPoint(cx1, cy1)
                ring.AddPoint(cx0, cy1)
                ring.AddPoint(cx0, cy0)
                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                # Buffer chunk slightly to ensure overlap is detected correctly
                if clip_geom and not poly.Intersects(clip_geom):
                    continue
                
                self.log(f"[{count}/{total}] Processing Chunk at X:{x} Y:{y}")
                
                # 3-pixel context buffer for slope kernel
                apron = i_res * 3
                win = [cx0 - apron, cy0 - apron, cx1 + apron, cy1 + apron]
                
                # Fetch and decode
                tmp = gdal.Warp('', vrt, format='MEM', outputBounds=win, xRes=i_res, yRes=i_res)
                r, g, b = [tmp.GetRasterBand(i).ReadAsArray().astype(np.float32) for i in (1,2,3)]
                elev = (r * 256.0 + g + b / 256.0) - 32768.0
                
                m_el = gdal.GetDriverByName('MEM').Create('', elev.shape[1], elev.shape[0], 1, gdal.GDT_Float32)
                m_el.SetGeoTransform(tmp.GetGeoTransform())
                m_el.SetProjection(tmp.GetProjection())
                m_el.GetRasterBand(1).WriteArray(elev)
                
                gdal.FillNodata(targetBand=m_el.GetRasterBand(1), maskBand=None, maxSearchDist=5, smoothingIterations=0)

                # Apply slight Blur/Median filter to elevation 
                # This softens "seams" where 10m and 30m data meet
                elev_healed = m_el.GetRasterBand(1).ReadAsArray()
                from scipy.ndimage import median_filter
                elev_healed = median_filter(elev_healed, size=3) # 3x3 kernel is enough
                m_el.GetRasterBand(1).WriteArray(elev_healed)

                # Run Slope - Return a NEW memory dataset instead of passing an existing one
                m_slp = gdal.DEMProcessing('', m_el, 'slope', format='MEM')
                
                # Average back into the target grid
                gdal.Warp(ds_final, m_slp, outputBounds=[cx0, cy0, cx1, cy1], resampleAlg='average')
                
                # CRITICAL: Explicitly clear objects to prevent the SIGKILL return
                m_slp = None
                m_el = None
                tmp = None
                del r, g, b, elev

    def _run_standard_process(self, vrt_path, ds_final, process_type):
        res = self.config['resolution']
        tmp = gdal.Warp('', vrt_path, format='MEM', xRes=res, yRes=res, 
                        outputBounds=self.config['bounds'], outputBoundsSRS="EPSG:4326", dstSRS="EPSG:3857")
        r, g, b = [tmp.GetRasterBand(i).ReadAsArray().astype(np.float32) for i in (1,2,3)]
        elev = (r * 256.0 + g + b / 256.0) - 32768.0
        
        if process_type == 'elevation':
            ds_final.GetRasterBand(1).WriteArray(elev)
        else:
            m_elev = gdal.GetDriverByName('MEM').Create('', r.shape[1], r.shape[0], 1, gdal.GDT_Float32)
            m_elev.SetGeoTransform(tmp.GetGeoTransform()); m_elev.SetProjection(tmp.GetProjection())
            m_elev.GetRasterBand(1).WriteArray(elev)
            gdal.DEMProcessing(ds_final, m_elev, process_type)

    def _post_process(self, src_path, output_path):
        mask_cfg = self.config.get('mask')
        clipping_path = self.config.get('clipping')
        final_src = src_path
        if clipping_path:
            self.log("Applying final clipping...")
            final_src = os.path.join(self.cache_dir, "final_clipped.tif")
            gdal.Warp(final_src, src_path, cutlineDSName=clipping_path, cropToCutline=True, dstNodata=-9999)

        ds = gdal.Open(final_src)
        data = ds.GetRasterBand(1).ReadAsArray()
        if mask_cfg:
            cond = np.ones(data.shape, dtype=bool)
            if 'min' in mask_cfg: cond &= (data >= mask_cfg['min'])
            if 'max' in mask_cfg: cond &= (data <= mask_cfg['max'])
            data = np.where(cond & (data != -9999), 1, 0).astype(np.uint8)

        if output_path.endswith('.gpkg'):
            self._save_as_vector(data, ds, output_path)
        else:
            self._save_raster(data, ds, output_path)

    def _save_raster(self, data, source_ds, path):
        driver = gdal.GetDriverByName("GTiff")
        dtype = gdal.GDT_Byte if data.dtype == np.uint8 else gdal.GDT_Float32
        out = driver.Create(path, source_ds.RasterXSize, source_ds.RasterYSize, 1, dtype, options=['COMPRESS=LZW'])
        out.SetProjection(source_ds.GetProjection()); out.SetGeoTransform(source_ds.GetGeoTransform())
        out.GetRasterBand(1).WriteArray(data)

    def _save_as_vector(self, data, source_ds, output_path):
        tmp_ds = gdal.GetDriverByName('MEM').Create('', source_ds.RasterXSize, source_ds.RasterYSize, 1, gdal.GDT_Byte)
        tmp_ds.SetProjection(source_ds.GetProjection()); tmp_ds.SetGeoTransform(source_ds.GetGeoTransform())
        tmp_ds.GetRasterBand(1).WriteArray(data)
        if os.path.exists(output_path): ogr.GetDriverByName("GPKG").DeleteDataSource(output_path)
        out = ogr.GetDriverByName("GPKG").CreateDataSource(output_path)
        srs = osr.SpatialReference(); srs.ImportFromWkt(source_ds.GetProjection())
        layer = out.CreateLayer("mask", srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("dn", ogr.OFTInteger))
        gdal.Polygonize(tmp_ds.GetRasterBand(1), tmp_ds.GetRasterBand(1), layer, 0)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: opendem <config.yml>")
        sys.exit(1)
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    app = OpenDEM(config_path)
    app.run()

if __name__ == "__main__":
    main()