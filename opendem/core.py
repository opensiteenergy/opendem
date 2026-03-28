import os
import math
import yaml
import shutil
import requests
import subprocess
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from osgeo import gdal, osr
from scipy.ndimage import gaussian_filter, median_filter, grey_opening, grey_closing

# Enable GDAL exceptions
gdal.UseExceptions()

class OpenDEMExporter:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_dir = os.path.abspath(self.config.get('cache_dir', './cache'))
        self.tile_cache = os.path.join(self.cache_dir, 'tiles')
        os.makedirs(self.tile_cache, exist_ok=True)
        
        self.zoom = self.config.get('zoom_level', 15)
        self.source_url_template = self.config['source']
        self.clipping_url = self.config.get('clipping')
        self.grid_size = float(self.config.get('grid_size', 10000.0))
        self.res = float(self.config.get('resolution', 1.0))
        self.buffer_m = float(self.config.get('buffer_meters', 500.0))
        
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

    def fetch_tile(self, x, y, z):
        tile_name = f"{z}_{x}_{y}.webp"
        local_path = os.path.join(self.tile_cache, tile_name)
        
        if local_path not in self.current_run_tiles:
            self.current_run_tiles.append(local_path)
        
        if not os.path.exists(local_path):
            url = self.source_url_template.replace('{z}', str(z)).replace('{x}', str(x)).replace('{y}', str(y))
            try:
                # Setting headers to look more like a browser to avoid 403s, 
                # though 404s usually mean the tile doesn't exist at that zoom/loc
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, timeout=10, headers=headers)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(response.content)
            except Exception:
                return None

        if not os.path.exists(local_path):
            return None

        src_ds = gdal.Open(local_path)
        if not src_ds: return None
        
        tw, th = src_ds.RasterXSize, src_ds.RasterYSize
        bounds = self.tile_to_mercator_bounds(x, y, z)
        pixel_size_x = (bounds[2] - bounds[0]) / float(tw)
        pixel_size_y = (bounds[3] - bounds[1]) / float(th)
        
        mem_driver = gdal.GetDriverByName('MEM')
        out_ds = mem_driver.Create('', tw, th, src_ds.RasterCount, gdal.GDT_Byte)
        out_ds.SetGeoTransform([bounds[0], pixel_size_x, 0, bounds[3], 0, -pixel_size_y])
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        out_ds.SetProjection(srs.ExportToWkt())
        
        for i in range(1, src_ds.RasterCount + 1):
            out_ds.GetRasterBand(i).WriteArray(src_ds.GetRasterBand(i).ReadAsArray())
            
        return out_ds

    def decode_terrarium(self, rgb_array):
        r, g, b = rgb_array[0].astype(np.float64), rgb_array[1].astype(np.float64), rgb_array[2].astype(np.float64)
        height = (r * 256.0 + g + b / 256.0) - 32768.0
        return median_filter(height, size=3)

    def advanced_dtm_filter(self, dtm_array):
        footprint_size = 30 
        temp = grey_closing(dtm_array, size=(footprint_size + 4, footprint_size + 4))
        cleaned = grey_opening(temp, size=(footprint_size, footprint_size))
        return gaussian_filter(cleaned, sigma=4)

    def calculate_slope(self, dtm_array, resolution):
        dy, dx = np.gradient(dtm_array, resolution)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    def cleanup_tiles(self):
        # We only cleanup if the config doesn't want to keep the cache
        if self.config.get('cleanup_cache', False):
            print(f"Cleaning up {len(self.current_run_tiles)} tiles...")
            for tile_path in self.current_run_tiles:
                if os.path.exists(tile_path):
                    try: os.remove(tile_path)
                    except: pass
        self.current_run_tiles = []

    def process_grid_cell(self, cell_bounds_3857, cell_id):
        try:
            buff_bounds = [
                cell_bounds_3857[0] - self.buffer_m, cell_bounds_3857[1] - self.buffer_m,
                cell_bounds_3857[2] + self.buffer_m, cell_bounds_3857[3] + self.buffer_m
            ]

            srs_3857 = osr.SpatialReference()
            srs_3857.ImportFromEPSG(3857)
            srs_4326 = osr.SpatialReference()
            srs_4326.ImportFromEPSG(4326)
            srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            tx_to_4326 = osr.CoordinateTransformation(srs_3857, srs_4326)

            c1 = tx_to_4326.TransformPoint(buff_bounds[0], buff_bounds[1])
            c2 = tx_to_4326.TransformPoint(buff_bounds[2], buff_bounds[3])
            
            start_x, start_y = self.deg2num(max(c1[1], c2[1]), min(c1[0], c2[0]), self.zoom)
            end_x, end_y = self.deg2num(min(c1[1], c2[1]), max(c1[0], c2[0]), self.zoom)
            
            tile_datasets = []
            for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
                for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                    print(f"{x}/{end_x}, {y}/{end_y}")
                    ds = self.fetch_tile(x, y, self.zoom)
                    if ds: tile_datasets.append(ds)

            if not tile_datasets: 
                return None

            vrt_ds = gdal.BuildVRT('', tile_datasets)
            buffered_ds = gdal.Warp('', vrt_ds, format='MEM', outputBounds=buff_bounds, 
                                    resampleAlg=gdal.GRIORA_Cubic, xRes=self.res, yRes=self.res, 
                                    dstSRS=srs_3857.ExportToWkt())

            rgb = buffered_ds.ReadAsArray()
            dtm_buff = self.advanced_dtm_filter(self.decode_terrarium(rgb))
            slope_buff = self.calculate_slope(dtm_buff, self.res)

            temp_slope_tif = os.path.join(self.cache_dir, f"temp_slope_{cell_id}.tif")
            temp_gpkg = os.path.join(self.cache_dir, f"temp_vec_{cell_id}.gpkg")
            
            drv = gdal.GetDriverByName('GTiff')
            ds = drv.Create(temp_slope_tif, buffered_ds.RasterXSize, buffered_ds.RasterYSize, 1, gdal.GDT_Float32)
            ds.SetGeoTransform(buffered_ds.GetGeoTransform())
            ds.SetProjection(buffered_ds.GetProjection())
            ds.GetRasterBand(1).WriteArray(slope_buff)
            ds.FlushCache()
            ds = None # Close file

            subprocess.run([
                "gdal_contour", "-p", "-amin", "slope_min", "-amax", "slope_max",
                "-fl", str(self.s_min), "-fl", str(self.s_max),
                "-off", "0", temp_slope_tif, temp_gpkg, "-f", "GPKG"
            ], check=True, capture_output=True)

            final_cell_gpkg = os.path.join(self.cache_dir, f"cell_{cell_id}.gpkg")
            subprocess.run([
                "ogr2ogr", "-f", "GPKG", final_cell_gpkg, temp_gpkg,
                "-clipsrc", str(cell_bounds_3857[0]), str(cell_bounds_3857[1]), 
                str(cell_bounds_3857[2]), str(cell_bounds_3857[3]),
                "-overwrite"
            ], check=True, capture_output=True)

            for f in [temp_slope_tif, temp_gpkg]:
                if os.path.exists(f): os.remove(f)
            
            return final_cell_gpkg
        except Exception as e:
            return None

    def generate_aoi_grid(self, aoi_gdf, grid_size, export_path=None):
        """
        Automatically generates a grid of polygons that intersect the AOI.
        
        Args:
            aoi_gdf: GeoDataFrame containing the clipping area (must be in EPSG:3857).
            grid_size: The size of each square cell in meters.
            export_path: Optional path to save a .gpkg for verification.
            
        Returns:
            GeoDataFrame of intersecting grid cells.
        """
        # 1. Get the total bounds
        minx, miny, maxx, maxy = aoi_gdf.total_bounds
        
        # 2. Create the coordinate arrays
        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)
        
        # 3. Create potential grid using list comprehension
        polygons = [
            box(x, y, x + grid_size, y + grid_size) 
            for x in x_coords 
            for y in y_coords
        ]
                
        # 4. Convert potential grid to a GeoDataFrame
        grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=aoi_gdf.crs)
        # Assign a unique ID to each cell for easier tracking in GIS
        grid_gdf['cell_id'] = range(len(grid_gdf))
        
        # 5. Spatial Join to filter active cells
        # 'inner' join ensures we only keep boxes that hit the AOI
        intersecting_grid = gpd.sjoin(grid_gdf, aoi_gdf, how="inner", predicate="intersects")
        
        # Clean up: Drop duplicates and remove join columns
        intersecting_grid = intersecting_grid.drop_duplicates(subset=['cell_id'])
        intersecting_grid = intersecting_grid[['cell_id', 'geometry']]
        
        print(f"Grid filtered: {len(grid_gdf)} potential -> {len(intersecting_grid)} active.")

        # 6. Optional Export for validation
        if export_path:
            print(f"Exporting validation grid to: {export_path}")
            intersecting_grid.to_file(export_path, driver="GPKG", layer="grid_cells")
        
        return intersecting_grid

    def run_grid_processing(self):

        # cell_bounds = [np.float64(-1523695.738230237), np.float64(7822634.741026822), np.float64(-1423695.738230237), np.float64(7922634.741026822)]
        # result_file = self.process_grid_cell(cell_bounds, "test")
        # exit()

        if not self.clipping_url:
            print("No clipping URL.")
            return

        print(f"Loading AOI: {self.clipping_url}")
        aoi_gdf = gpd.read_file(self.clipping_url)
        if aoi_gdf.crs.to_epsg() != 3857:
            aoi_gdf = aoi_gdf.to_crs(epsg=3857)

        active_cells = self.generate_aoi_grid(aoi_gdf, self.grid_size, "active_cells.gpkg")

        cell_count = 0
        cell_files = []
        try:
            for idx, row in active_cells.iterrows():
                cell_count += 1
                cell_bounds = row.geometry.bounds

                print(f"Processing active cell {cell_count} of {len(active_cells)} {cell_bounds}")

                result_file = self.process_grid_cell(cell_bounds, f"{cell_count}")

                if result_file and os.path.exists(result_file):
                    cell_files.append(result_file)

            if cell_files:
                print(f"Merging {len(cell_files)} cells...")
                final_output = os.path.join(self.cache_dir, "final_analysis_grid.gpkg")
                subprocess.run(["ogrmerge.py", "-single", "-o", final_output] + cell_files + ["-overwrite_ds"], check=True)
                print(f"Done: {final_output}")
                for f in cell_files:
                    if os.path.exists(f): os.remove(f)
            else:
                print("No steep areas found in any cells.")

        finally:
            self.cleanup_tiles()

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