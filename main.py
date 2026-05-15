from src.extract.scraper import extract_adzuna_jobs
from src.transform.processor import process_raw_jobs
from src.load.database import load_data_to_duckdb

if __name__ == "__main__":
    extract_adzuna_jobs() 
    process_raw_jobs()    
    
    # A nova etapa:
    load_data_to_duckdb()