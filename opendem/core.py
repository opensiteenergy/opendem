import os
import math
import yaml
import shutil
import requests
import subprocess
import sys
import numpy as np
import geopandas as gpd
import threading
import concurrent.futures
from urllib.parse import urlparse
from shapely.geometry import box
from osgeo import gdal, osr
from scipy.ndimage import (
    gaussian_filter, 
    median_filter,
    uniform_filter,
    grey_closing, 
    grey_opening
)

# # Disable GDAL's Persistent Auxiliary Metadata (PAM) to prevent .aux.xml files
# gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
# Enable GDAL exceptions
gdal.UseExceptions()

class OpenDEMExporter:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.failed_tiles = set()
        self.multithread_lock = threading.Lock()
        self.cache_dir = os.path.abspath(self.config.get('cache_dir', './cache'))
        self.tile_cache = os.path.join(self.cache_dir, 'tiles')
        os.makedirs(self.tile_cache, exist_ok=True)
        
        self.zoom = self.config.get('zoom_level', 15)
        self.source_url_template = self.config['source']
        self.clipping = self.config.get('clipping')
        self.output = self.config.get('output')
        self.grid_size = float(self.config.get('grid_size', 20000.0))
        self.res = float(self.config.get('resolution', 1.0))
        self.buffer_m = float(self.config.get('buffer_meters', 500.0))
        
        self.min_width = float(self.config.get('min_polygon_width', 2.0)) # Threads thinner than this are removed

        self.s_min = float(self.config['mask']['min'])
        self.s_max = float(self.config['mask']['max'])
        
        self.current_run_tiles = []

    def deg2num(self, lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    def tile_to_mercator_bounds(self, x, y, z):
        world_size = 20037508.342789244 * 2
        res = world_size / (2**z)
        origin_shift = world_size / 2.0
        minx = x * res - origin_shift
        maxx = (x + 1) * res - origin_shift
        maxy = origin_shift - y * res
        miny = origin_shift - (y + 1) * res
        return [minx, miny, maxx, maxy]

    def fetch_tile(self, x, y, z, target_x=None, target_y=None, target_z=None):
        """
        Fetches a tile at zoom level z. If not found, recurses up to find a parent
        and upsamples using Cubic Spline interpolation to prevent slope artifacts.
        """
        # Initialize targets on first call
        if target_x is None: target_x = x
        if target_y is None: target_y = y
        if target_z is None: target_z = z
            
        target_tile_name = f"{target_z}_{target_x}_{target_y}.webp"
        target_local_path = os.path.join(self.tile_cache, target_tile_name)
        
        with self.multithread_lock:
            if target_local_path not in self.current_run_tiles:
                self.current_run_tiles.append(target_local_path)
        
        # --- PHASE 1: DATA ACQUISITION ---
        current_tile_name = f"{z}_{x}_{y}.webp"
        current_local_path = os.path.join(self.tile_cache, current_tile_name)
        
        if not os.path.exists(current_local_path):
            url = self.source_url_template.replace('{z}', str(z)).replace('{x}', str(x)).replace('{y}', str(y))
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                failed_cache = False

                # with self.multithread_lock:
                #     if url in self.failed_tiles: failed_cache = True

                if failed_cache:
                    response = {"status_code": 404}
                else:
                    response = requests.get(url, timeout=10, headers=headers)

                if response.status_code == 404:
                    # with self.multithread_lock:
                    #     if url not in self.failed_tiles:
                    #         self.failed_tiles.add(url)
                        
                    if z <= 10: return None
                    # Recurse up to find parent data
                    # print(f" --> Unable to retrieve tile for {x} {y} {z} - moving up to {z - 1}")
                    return self.fetch_tile(x // 2, y // 2, z - 1, target_x, target_y, target_z)

                print(f" --> Successfully retrieved tile for {x} {y} {z}")

                response.raise_for_status()
                with self.multithread_lock:
                    with open(current_local_path, 'wb') as f:
                        f.write(response.content)
            except Exception:
                return None

        # --- PHASE 2: PROCESSING ---

        src_ds = None
        with self.multithread_lock:
            src_ds = gdal.Open(current_local_path)
            if not src_ds:
                return None
        
            src_w, src_h = src_ds.RasterXSize, src_ds.RasterYSize

            # FIX: Assign georeference to the source tile so ReprojectImage knows its location
            tile_bounds = self.tile_to_mercator_bounds(x, y, z)
            src_ds.SetGeoTransform([tile_bounds[0], (tile_bounds[2]-tile_bounds[0])/src_w, 0, tile_bounds[3], 0, -(tile_bounds[3]-tile_bounds[1])/src_h])
            srs_src = osr.SpatialReference(); srs_src.ImportFromEPSG(3857)
            src_ds.SetProjection(srs_src.ExportToWkt())

            out_w, out_h = 512, 512
            
            bounds = self.tile_to_mercator_bounds(target_x, target_y, target_z)
            pixel_size_x = (bounds[2] - bounds[0]) / float(out_w)
            pixel_size_y = (bounds[3] - bounds[1]) / float(out_h)
            
            mem_driver = gdal.GetDriverByName('MEM')
            out_ds = mem_driver.Create('', out_w, out_h, src_ds.RasterCount, gdal.GDT_Byte)
            out_ds.SetGeoTransform([bounds[0], pixel_size_x, 0, bounds[3], 0, -pixel_size_y])
            
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            out_ds.SetProjection(srs.ExportToWkt())

            # --- PHASE 3: INTERPOLATION (The Anti-Artifact Fix) ---
            # We switch from Bilinear to Cubicspline. 
            # Bilinear is C0 continuous (values match), but not C1 continuous (slopes don't match).
            # Cubicspline provides C1 continuity, removing the "grid" lines in slope analysis.
            
            if z == target_z:
                gdal.ReprojectImage(src_ds, out_ds, None, None, gdal.GRA_CubicSpline)
            else:
                scale = 2 ** (target_z - z)
                win_w, win_h = src_w / scale, src_h / scale
                win_x, win_y = (target_x % scale) * win_w, (target_y % scale) * win_h
                
                for i in range(1, src_ds.RasterCount + 1):
                    data = src_ds.GetRasterBand(i).ReadAsArray(
                        int(win_x), int(win_y), int(win_w), int(win_h), 
                        buf_xsize=out_w, buf_ysize=out_h, 
                        resample_alg=gdal.GRIORA_CubicSpline
                    )
                    if data is not None:
                        out_ds.GetRasterBand(i).WriteArray(data)

            # --- PHASE 4: PERSISTENCE ---
            if not os.path.exists(target_local_path):
                webp_driver = gdal.GetDriverByName('WEBP')
                if webp_driver:
                    # Maintain LOSSLESS=YES to ensure no further artifacts are introduced by compression
                    saved_ds = webp_driver.CreateCopy(
                        target_local_path, 
                        out_ds, 
                        strict=0, 
                        options=["LOSSLESS=YES"]
                    )
                    saved_ds = None 
                    
            return out_ds
        
    def filter_slender_polygons(self, gpkg_path):
        """
        Removes 'threads' by performing a Morphological Opening on the geometry.
        If a part of a polygon is thinner than 2 * self.min_width, it will be erased.
        """
        try:
            gdf = gpd.read_file(gpkg_path)
            if gdf.empty: return None
            
            # Step 1: Erosion (Negative Buffer)
            # This makes thin parts disappear
            dist = self.min_width / 2.0
            gdf['geometry'] = gdf.geometry.buffer(-dist)
            
            # Remove empty geometries resulting from erosion
            gdf = gdf[~gdf.geometry.is_empty]
            
            if gdf.empty: return None

            # Step 2: Dilation (Positive Buffer)
            # This brings the 'fat' parts back to their original size
            gdf['geometry'] = gdf.geometry.buffer(dist)
            
            # Step 3: Cleanup
            # Explode multi-polygons that might have been split by the process
            gdf = gdf.explode(index_parts=True).reset_index(drop=True)
            
            # Re-save the filtered data
            gdf.to_file(gpkg_path, driver="GPKG")
            return gpkg_path
        except Exception as e:
            print(f"Filtering error: {e}")
            return gpkg_path

    def batch_filter_cache(self):
        """Iterates through cached cell GPKGs and filters them."""
        pattern = os.path.join(self.cache_dir, "cell_*.gpkg")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No cached cell files found in {self.cache_dir}")
            return

        print(f"Found {len(files)} cached cells. Applying min_width filter: {self.min_width}m...")
        filtered_files = []
        for f in files:
            result = self.filter_slender_polygons(f)
            if result:
                filtered_files.append(result)
        
        if filtered_files:
            print(f"Re-merging {len(filtered_files)} filtered cells into {self.output}...")
            subprocess.run(["ogrmerge.py", "-single", "-o", self.output] + filtered_files + ["-overwrite_ds"], check=True)
            print("Done.")
        
    def decode_terrarium(self, rgb_array):
        if rgb_array.shape[0] < 3:
            return np.zeros((rgb_array.shape[1], rgb_array.shape[2]))
        r, g, b = rgb_array[0].astype(np.float64), rgb_array[1].astype(np.float64), rgb_array[2].astype(np.float64)
        height = (r * 256.0 + g + b / 256.0) - 32768.0
        return median_filter(height, size=3)

    # def advanced_dtm_filter(self, dtm_array):
    #     footprint_size = 30 
    #     temp = grey_closing(dtm_array, size=(footprint_size + 4, footprint_size + 4))
    #     cleaned = grey_opening(temp, size=(footprint_size, footprint_size))
    #     return gaussian_filter(cleaned, sigma=4)

    def advanced_dtm_filter(self, dtm_array):
        """
        Tuned pipeline to eliminate 1m-quantization 'rings' while 
        preserving the general terrain shape.
        """
        # 1. INITIAL BOX BLUR (Size 3)
        # This slightly softens the 'pixel-perfect' steps so the median 
        # filter doesn't treat them as valid signal.
        smeared = uniform_filter(dtm_array, size=3)
        
        # 2. MEDIAN FILTER (Size 5) - THE KEY STEP
        # This is non-linear. It looks at a 5x5 window and picks the middle value.
        # It is excellent at removing the 'terracing' effect that causes slope rings.
        denoised = median_filter(smeared, size=5)
        
        # 3. INCREASED GAUSSIAN BLUR (Sigma 6-8)
        # Now that the 'steps' are mathematically broken by the median filter, 
        # the Gaussian blur can create a truly smooth surface for slope calculation.
        # Sigma 7 is a good 'sweet spot' for 1-2m resolution data.
        return gaussian_filter(denoised, sigma=7)

    def calculate_slope(self, dtm_array, resolution):
        dy, dx = np.gradient(dtm_array, resolution)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    def cleanup_tiles(self):
        for tile_path in self.current_run_tiles:
            tile_metadata = tile_path + ".aux.xml"
            if os.path.exists(tile_path):
                try: os.remove(tile_path)
                except: pass
            if os.path.exists(tile_metadata):
                try: os.remove(tile_metadata)
                except: pass
        self.current_run_tiles = []

    def fetch_all_tiles(self, x_range, y_range):
        """
        Replaces the nested loop with a parallel thread pool to speed up tile downloads.
        """
        tile_datasets = []
        
        # Create a list of all (x, y) pairs to download
        coords = [(x, y) for x in x_range for y in y_range]
        
        # Use ThreadPoolExecutor for I/O bound network requests
        # max_workers=10 is a safe starting point; increase if the server allows
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # We wrap the call in a lambda or a small helper to pass self.zoom
            future_to_tile = {
                executor.submit(self.fetch_tile, x, y, self.zoom): (x, y) 
                for x, y in coords
            }
            
            for future in concurrent.futures.as_completed(future_to_tile):
                coord = future_to_tile[future]
                try:
                    ds = future.result()
                    if ds:
                        tile_datasets.append(ds)
                except Exception as e:
                    print(f"Tile {coord} generated an exception: {e}")

    def fetch_local_tile(self, x, y, z):
        """
        Synchronously loads a tile from the local cache.
        Returns a GDAL dataset or None if the file is missing/corrupt.
        """

        current_tile_name = f"{z}_{x}_{y}.webp"
        current_local_path = os.path.join(self.tile_cache, current_tile_name)
        
        if not os.path.exists(current_local_path):
            return None
            
        try:
            # Open the local file. Using GA_ReadOnly for safety.
            ds = gdal.Open(current_local_path, gdal.GA_ReadOnly)
            return ds
        except Exception as e:
            print(f"Failed to load local tile {x}, {y}: {e}")
            return None

    def process_grid_cell(self, cell_bounds_3857, cell_id):
        """
        Processes an individual grid cell. 
        Includes robust coordinate conversion and check for empty data.
        """
        try:
            final_cell_gpkg = os.path.join(self.cache_dir, f"cell_{cell_id}.gpkg")
            if os.path.exists(final_cell_gpkg): return final_cell_gpkg
            
            # Ensure we are working with standard floats, not numpy types
            minx, miny, maxx, maxy = [float(b) for b in cell_bounds_3857]
            
            buff_bounds = [
                minx - self.buffer_m, miny - self.buffer_m,
                maxx + self.buffer_m, maxy + self.buffer_m
            ]

            srs_3857 = osr.SpatialReference()
            srs_3857.ImportFromEPSG(3857)
            srs_4326 = osr.SpatialReference()
            srs_4326.ImportFromEPSG(4326)
            srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            tx_to_4326 = osr.CoordinateTransformation(srs_3857, srs_4326)

            # Transform corners to 4326 for tile fetching
            c1 = tx_to_4326.TransformPoint(buff_bounds[0], buff_bounds[1])
            c2 = tx_to_4326.TransformPoint(buff_bounds[2], buff_bounds[3])
            
            # Identify tile range
            lats = [c1[1], c2[1]]
            lons = [c1[0], c2[0]]
            start_x, start_y = self.deg2num(max(lats), min(lons), self.zoom)
            end_x, end_y = self.deg2num(min(lats), max(lons), self.zoom)
            
            tile_datasets = []
            x_range = range(min(start_x, end_x), max(start_x, end_x) + 1)
            y_range = range(min(start_y, end_y), max(start_y, end_y) + 1)
            
            self.fetch_all_tiles(x_range, y_range)

            for x in x_range:
                for y in y_range:
                    ds = self.fetch_local_tile(x, y, self.zoom)
                    if ds: 
                        tile_datasets.append(ds)

            if not tile_datasets: 
                print(f"Cell {cell_id}: No source tiles found in coverage area.")
                return None

            # Build virtual mosaic of tiles
            vrt_ds = gdal.BuildVRT('', tile_datasets)
            
            # Warp to the exact grid cell (plus buffer) in 3857
            buffered_ds = gdal.Warp('', vrt_ds, format='MEM', outputBounds=buff_bounds, 
                                    resampleAlg=gdal.GRIORA_Cubic, xRes=self.res, yRes=self.res, 
                                    dstSRS=srs_3857.ExportToWkt())

            rgb = buffered_ds.ReadAsArray()
            
            # Robustness check: if the array is empty or zeroed (common on edges)
            if rgb is None or np.all(rgb == 0):
                print(f"Cell {cell_id}: Raster data is empty/zero.")
                return None

            # dtm_buff = self.decode_terrarium(rgb)
            dtm_buff = self.advanced_dtm_filter(self.decode_terrarium(rgb))
            slope_buff = self.calculate_slope(dtm_buff, self.res)

            temp_dtm_tif = os.path.join(self.cache_dir, f"temp_dtm_{cell_id}.tif")
            temp_slope_tif = os.path.join(self.cache_dir, f"temp_slope_{cell_id}.tif")
            temp_gpkg = os.path.join(self.cache_dir, f"temp_vec_{cell_id}.gpkg")

            drv = gdal.GetDriverByName('GTiff')

            ds_dtm = drv.Create(temp_dtm_tif, buffered_ds.RasterXSize, buffered_ds.RasterYSize, 1, gdal.GDT_Float32)
            ds_dtm.SetGeoTransform(buffered_ds.GetGeoTransform())
            ds_dtm.SetProjection(buffered_ds.GetProjection())
            ds_dtm.GetRasterBand(1).WriteArray(dtm_buff)
            ds_dtm.FlushCache()
            ds_dtm = None

            ds = drv.Create(temp_slope_tif, buffered_ds.RasterXSize, buffered_ds.RasterYSize, 1, gdal.GDT_Float32)
            ds.SetGeoTransform(buffered_ds.GetGeoTransform())
            ds.SetProjection(buffered_ds.GetProjection())
            ds.GetRasterBand(1).WriteArray(slope_buff)
            ds.FlushCache()
            ds = None 

            # Vectorize slope into polygons
            subprocess.run([
                "gdal_contour", "-p", "-amin", "slope_min", "-amax", "slope_max",
                "-fl", str(self.s_min), "-fl", str(self.s_max),
                "-off", "0", temp_slope_tif, temp_gpkg, "-f", "GPKG"
            ], check=True, capture_output=True)

            # Final clip to the unbuffered grid cell boundaries
            subprocess.run([
                "ogr2ogr", "-f", "GPKG", final_cell_gpkg, temp_gpkg,
                "-clipsrc", str(minx), str(miny), str(maxx), str(maxy),
                "-nlt", "MULTIPOLYGON", "-overwrite"
            ], check=True, capture_output=True)

            # Cleanup intermediate files
            for f in [temp_slope_tif, temp_gpkg, temp_dtm_tif]:
                if os.path.exists(f): os.remove(f)
            
            return final_cell_gpkg
            
        except Exception as e:
            print(f"Error processing cell {cell_id}: {e}")
            return None

    def generate_aoi_grid(self, aoi_gdf, grid_size, export_path=None):
        """
        Generates a grid of squares covering the AOI and filters for intersection.
        
        Common pitfalls fixed:
        1. CRS Alignment: Ensures the grid is born in the same CRS as the AOI.
        2. Coordinate Order: Nested loops can flip X/Y expectations in logs.
        3. Attribute Cleanup: SJOIN often brings in extra columns that clutter the format.
        """
        # 1. Get bounds in the native CRS of the AOI
        # If this is EPSG:3857, miny should be ~6.7M for Bristol.
        # If it's 7.8M, your input 'aoi_gdf' is likely in the wrong place.
        minx, miny, maxx, maxy = aoi_gdf.total_bounds
        
        # Create ranges for the grid
        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)
        
        polygons = []
        for x in x_coords:
            for y in y_coords:
                # Create the square
                polygons.append(box(x, y, x + grid_size, y + grid_size))
                
        # 2. Create Grid GeoDataFrame
        grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=aoi_gdf.crs)
        grid_gdf['cell_id'] = range(len(grid_gdf))
        
        # 3. Spatial Join
        # 'inner' ensures we only keep cells that touch the AOI
        intersecting_grid = gpd.sjoin(
            grid_gdf, 
            aoi_gdf[['geometry']], # Only pass geometry to avoid column name collisions
            how="inner", 
            predicate="intersects"
        )
        
        # 4. Cleanup
        # drop_duplicates is necessary because one grid cell might touch multiple AOI features
        intersecting_grid = intersecting_grid.drop_duplicates(subset=['cell_id'])
        
        # Ensure index is clean and only necessary columns exist
        intersecting_grid = intersecting_grid[['cell_id', 'geometry']].reset_index(drop=True)
        
        print(f"Grid filtered: {len(grid_gdf)} potential -> {len(intersecting_grid)} active.")

        if export_path:
            # Exporting to GPKG for visualization in QGIS/ArcGIS
            intersecting_grid.to_file(export_path, driver="GPKG", layer="grid_cells")
        
        return intersecting_grid
        
    def run_grid_processing(self):

        if not self.clipping:
            print("No clipping URL.")
            return

        print(f"Loading AOI: {self.clipping}")
        aoi_gdf = gpd.read_file(self.clipping)
        if aoi_gdf.crs.to_epsg() != 3857:
            aoi_gdf = aoi_gdf.to_crs(epsg=3857)

        active_cells = self.generate_aoi_grid(aoi_gdf, self.grid_size, "active_cells.gpkg")

        cell_files = []
        cell_count = 0
        for idx, row in active_cells.iterrows():
            cell_count += 1
            cell_id = f"{cell_count}"
            cell_bounds = row.geometry.bounds # (minx, miny, maxx, maxy)

            print(f"Processing cell {cell_id}, fid {idx + 1}: {cell_count}/{len(active_cells)} {cell_bounds}")

            result_file = self.process_grid_cell(cell_bounds, cell_id)
            self.cleanup_tiles()

            if result_file and os.path.exists(result_file):
                cell_files.append(result_file)

        if cell_files:
            intermediate_merged = os.path.join(self.cache_dir, "temp_merged_unclipped.gpkg")
            intermediate_exploded = os.path.join(self.cache_dir, "temp_merged_clipped_exploded.gpkg")
            intermediate_file = intermediate_merged
            if not os.path.exists(intermediate_merged):
                print(f"Merging {len(cell_files)} cells...")
                subprocess.run(["ogrmerge.py", "-single", "-o", intermediate_merged] + cell_files + ["-overwrite_ds"], check=True)

            # 1. Flexible check for local file OR URL
            is_clipping_str = isinstance(self.clipping, str)
            is_clipping_url = is_clipping_str and urlparse(self.clipping).scheme in ('http', 'https', 'ftp')
            is_clipping_file = is_clipping_str and os.path.exists(self.clipping)
            if is_clipping_url or is_clipping_file:
                intermediate_clipped = os.path.join(self.cache_dir, "temp_merged_clipped.gpkg")

                if not os.path.exists(intermediate_clipped):
                    print(f"Clipping merged file using source: {self.clipping}...")

                    # 2. Build the command
                    # GDAL handles URLs natively via /vsicurl/ or direct string passing
                    cmd = [
                        "ogr2ogr",
                        "-f", "GPKG",
                        intermediate_clipped,
                        intermediate_merged,
                        "-clipsrc", self.clipping,
                        "-nlt", "PROMOTE_TO_MULTI"
                    ]

                    # 3. Optional: GDAL config for better URL handling (timeouts, etc.)
                    env = os.environ.copy()
                    if is_clipping_url:
                        env["GDAL_HTTP_RETRY_COUNT"] = "3"
                        env["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".gpkg,.shp,.json,.geojson"

                    subprocess.run(cmd, env=env, check=True)
                    
                    # Clean up the intermediate merged file
                    if os.path.exists(intermediate_merged):
                        os.remove(intermediate_merged)
                    
                    print(f"Done! Final clipped output saved to: {intermediate_clipped}")

                intermediate_file = intermediate_clipped

            if not os.path.exists(intermediate_exploded):
                print(f"Exploding file: {intermediate_file} to {intermediate_exploded}")
                gdf = gpd.read_file(intermediate_file)
                gdf_dumped = gdf.explode()
                gdf_dumped.to_file(intermediate_exploded, driver="GPKG")

            exit()
            shutil.move(intermediate_exploded, self.output)
            print(f"Final output moved to: {self.output}")

            # # Clean up intermediate files
            # for f in cell_files:
            #     if os.path.exists(f): os.remove(f)

        else:
            print("No valid output generated for any cells.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python opendem_exporter.py <config.yml>")
        sys.exit(1)
    
    try:
        exporter = OpenDEMExporter(sys.argv[1])
        exporter.run_grid_processing()
    except Exception as e:
        print(f"Fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()