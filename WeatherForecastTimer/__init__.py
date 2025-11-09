import os
import psycopg2
import psycopg2.extras
import urllib3
import pandas as pd
import time
from datetime import datetime
import openmeteo_requests
import requests_cache
import requests
from retry_requests import retry
from typing import Any, List, Tuple, Optional, cast
import logging
from contextlib import contextmanager
import azure.functions as func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Get DATABASE_URL from environment variable (set in Azure Function App Settings)
DATABASE_URL = os.environ.get("DATABASE_URL")

# Setup Open-Meteo client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.1)
# requests_cache.CachedSession is runtime-compatible with requests.Session, but static type checkers may disagree;
# use cast to satisfy the function signature expected by openmeteo_requests.Client.
openmeteo = openmeteo_requests.Client(session=cast(Any, retry_session))

# All weather variables
ALL_VARIABLES = [
    "temperature_2m","relative_humidity_2m","dew_point_2m","apparent_temperature",
    "precipitation_probability","precipitation","rain","showers","snowfall","snow_depth",
    "weather_code","pressure_msl","surface_pressure","cloud_cover","cloud_cover_low",
    "cloud_cover_mid","cloud_cover_high","visibility","evapotranspiration",
    "et0_fao_evapotranspiration","vapour_pressure_deficit","temperature_180m",
    "temperature_120m","temperature_80m","wind_gusts_10m","wind_direction_180m",
    "wind_direction_120m","wind_direction_80m","wind_direction_10m","wind_speed_180m",
    "wind_speed_120m","wind_speed_80m","wind_speed_10m"
]

@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic cleanup"""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    finally:
        if conn:
            conn.close()

def cockroachdb_retry_transaction(func, max_retries=5):
    """Retry wrapper for CockroachDB serializable transaction conflicts"""
    last_exception = None
    for attempt in range(max_retries):
        try:
            result = func()
            if result is None:
                # Always return a tuple to avoid NoneType errors
                return (0, 0)
            return result
        except psycopg2.Error as e:
            error_msg = str(e).lower()
            if ("retry" in error_msg and "serializable" in error_msg) or "restart transaction" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.1 + (attempt * 0.05)
                    logger.warning(f"Transaction conflict, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded for transaction")
                    last_exception = e
                    break
            else:
                raise e
    if last_exception:
        raise last_exception
    return (0, 0)

def fetch_weather_forecast(lat: float, lon: float) -> Optional[pd.DataFrame]:
    """Optimized weather forecast fetching"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ALL_VARIABLES,
        "models": "best_match",
        "forecast_days": 14,
        "timezone": "auto"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        if not responses:
            return None
            
        response = responses[0]
        hourly = response.Hourly()
        
        if not hourly:
            return None

        time_start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
        time_end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
        time_interval = pd.Timedelta(seconds=hourly.Interval())
        
        dates = pd.date_range(start=time_start, end=time_end, freq=time_interval, inclusive="left")
        hourly_data: dict = {}
        hourly_data["date"] = dates

        try:
            variables_length = hourly.VariablesLength() if hasattr(hourly, 'VariablesLength') else len(ALL_VARIABLES)
            
            for i in range(min(len(ALL_VARIABLES), variables_length)):
                var_name = ALL_VARIABLES[i]
                try:
                    variable = hourly.Variables(i)
                    if variable is not None:
                        values_getter = getattr(variable, 'ValuesAsNumpy', None)
                        if callable(values_getter):
                            try:
                                values = values_getter()
                                if values is None:
                                    data_list = None
                                else:
                                    # Try to convert to list, handling different types
                                    try:
                                        # First try numpy array tolist()
                                        if hasattr(values, 'tolist') and callable(getattr(values, 'tolist')):
                                            data_list = values.tolist()
                                        # Then try regular iteration
                                        elif hasattr(values, '__iter__') and not isinstance(values, (str, bytes)):
                                            data_list = list(values)
                                        else:
                                            data_list = None
                                    except Exception:
                                        data_list = None
                                if not isinstance(data_list, list) or len(data_list) != len(dates):
                                    hourly_data[var_name] = pd.Series([None] * len(dates), dtype='float64')
                                else:
                                    hourly_data[var_name] = pd.Series(data_list, dtype='float64')
                            except Exception:
                                hourly_data[var_name] = pd.Series([None] * len(dates), dtype='float64')
                        else:
                            hourly_data[var_name] = pd.Series([None] * len(dates), dtype='float64')
                    else:
                        hourly_data[var_name] = pd.Series([None] * len(dates), dtype='float64')
                except Exception:
                    hourly_data[var_name] = pd.Series([None] * len(dates), dtype='float64')

        except Exception as e:
            logger.error(f"Error processing variables for ({lat}, {lon}): {e}")
            return None

        df = pd.DataFrame(hourly_data)
        df = df.dropna(subset=['date'])
        
        return df if not df.empty else None
        
    except Exception as e:
        logger.error(f"API request failed for ({lat}, {lon}): {e}")
        return None

def update_weather_data_for_site(site_id: int, forecast_df: pd.DataFrame) -> Tuple[bool, str]:
    """Update weather data for a single site with optimized DELETE + INSERT"""
    
    def execute_update():
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            delete_query = "DELETE FROM public.hourly_weather WHERE site_id = %s AND date >= CURRENT_DATE;"
            cur.execute(delete_query, (site_id,))
            deleted_count = cur.rowcount
            
            insert_query = """
                INSERT INTO public.hourly_weather (
                    site_id, date, temperature_2m, relative_humidity_2m, dew_point_2m,
                    apparent_temperature, precipitation_probability, precipitation,
                    rain, showers, snowfall, snow_depth, weather_code, pressure_msl,
                    surface_pressure, cloud_cover, cloud_cover_low, cloud_cover_mid,
                    cloud_cover_high, visibility, evapotranspiration, et0_fao_evapotranspiration,
                    vapour_pressure_deficit, temperature_180m, temperature_120m, temperature_80m,
                    wind_gusts_10m, wind_direction_180m, wind_direction_120m, wind_direction_80m,
                    wind_direction_10m, wind_speed_180m, wind_speed_120m, wind_speed_80m,
                    wind_speed_10m
                ) VALUES %s
            """
            
            insert_data = []
            for _, row in forecast_df.iterrows():
                if pd.isna(row['date']):
                    continue
                    
                values = (
                    site_id, row['date'], row.get('temperature_2m'), row.get('relative_humidity_2m'),
                    row.get('dew_point_2m'), row.get('apparent_temperature'), row.get('precipitation_probability'),
                    row.get('precipitation'), row.get('rain'), row.get('showers'), row.get('snowfall'),
                    row.get('snow_depth'), row.get('weather_code'), row.get('pressure_msl'),
                    row.get('surface_pressure'), row.get('cloud_cover'), row.get('cloud_cover_low'),
                    row.get('cloud_cover_mid'), row.get('cloud_cover_high'), row.get('visibility'),
                    row.get('evapotranspiration'), row.get('et0_fao_evapotranspiration'),
                    row.get('vapour_pressure_deficit'), row.get('temperature_180m'), row.get('temperature_120m'),
                    row.get('temperature_80m'), row.get('wind_gusts_10m'), row.get('wind_direction_180m'),
                    row.get('wind_direction_120m'), row.get('wind_direction_80m'), row.get('wind_direction_10m'),
                    row.get('wind_speed_180m'), row.get('wind_speed_120m'), row.get('wind_speed_80m'),
                    row.get('wind_speed_10m')
                )
                insert_data.append(values)
            
            if insert_data:
                psycopg2.extras.execute_values(
                    cur, insert_query, insert_data,
                    template=None, page_size=200
                )
                
            conn.commit()
            return deleted_count, len(insert_data)
    
    try:
        deleted_count, inserted_count = cockroachdb_retry_transaction(execute_update)
        return True, f"Deleted {deleted_count}, inserted {inserted_count} records"
    except Exception as e:
        return False, str(e)

def process_single_site(site_data: Tuple[int, float, float]) -> Tuple[int, bool, str]:
    """Process a single site's weather data"""
    site_id, lat, lon = site_data
    
    try:
        forecast_df = fetch_weather_forecast(lat, lon)
        
        if forecast_df is None or forecast_df.empty:
            return site_id, False, "No forecast data available"
        
        success, message = update_weather_data_for_site(site_id, forecast_df)
        return site_id, success, message
        
    except Exception as e:
        logger.error(f"Error processing site_id={site_id}: {e}")
        return site_id, False, str(e)

def process_all_sites_optimized(sites: List[Tuple[int, float, float]]) -> List[Tuple[int, bool, str]]:
    """Process all sites with optimized batching for CockroachDB"""
    results = []
    total_sites = len(sites)
    batch_size = 5
    
    for i in range(0, total_sites, batch_size):
        batch = sites[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_sites + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} sites)")
        
        for j, site_data in enumerate(batch):
            site_id, lat, lon = site_data
            logger.info(f"  Processing site_id={site_id} at ({lat}, {lon})")
            
            result = process_single_site(site_data)
            results.append(result)
            
            site_id, success, message = result
            if success:
                logger.info(f"  ✓ Site {site_id}: {message}")
            else:
                logger.error(f"  ✗ Site {site_id}: {message}")
            
            if j < len(batch) - 1:
                time.sleep(0.1)
        
        completed = min(i + batch_size, total_sites)
        logger.info(f"Progress: {completed}/{total_sites} sites processed ({(completed/total_sites)*100:.1f}%)")
        
        if i + batch_size < total_sites:
            time.sleep(0.5)
    
    return results

def main(timer: func.TimerRequest) -> None:
    """Azure Function entry point"""
    start_time = datetime.now()
    
    if mytimer.past_due:
        logger.info('The timer is past due!')
    
    logger.info("Starting weather forecast update...")
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT site_id, latitude, longitude FROM public.ski_sites WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY site_id;")
            sites = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch sites from database: {e}")
        return
    
    logger.info(f"Found {len(sites)} sites to fetch forecasts for.")
    
    results = process_all_sites_optimized(sites)
    
    successful_updates = sum(1 for _, success, _ in results if success)
    failed_updates = len(results) - successful_updates
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EXECUTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total execution time: {duration}")
    logger.info(f"Sites processed: {len(sites)}")
    logger.info(f"Successful updates: {successful_updates}")
    logger.info(f"Failed updates: {failed_updates}")
    logger.info(f"Success rate: {(successful_updates/len(sites)*100):.1f}%")
    
    if len(sites) > 0:
        logger.info(f"Average time per site: {duration.total_seconds()/len(sites):.2f} seconds")
    
    if failed_updates > 0:
        logger.info(f"\nFirst 5 failed sites:")
        failed_count = 0
        for site_id, success, message in results:
            if not success and failed_count < 5:
                logger.info(f"  Site {site_id}: {message[:100]}...")
                failed_count += 1
    
    
    logger.info(f"Function executed successfully. Processed {len(sites)} sites with {successful_updates} successful updates in {duration.total_seconds():.2f} seconds.")
