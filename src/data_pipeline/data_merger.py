import pandas as pd
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class DataMerger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge_datasets(self, google_play_df: pd.DataFrame, app_store_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Google Play and App Store datasets"""
        
        self.logger.info(f"Merging datasets: {len(google_play_df)} Google Play + {len(app_store_df)} App Store apps")
        
        # Ensure both DataFrames have 'Store' column
        if 'Store' not in google_play_df.columns:
            google_play_df = google_play_df.copy()
            google_play_df['Store'] = 'Google Play'
        
        if not app_store_df.empty and 'Store' not in app_store_df.columns:
            app_store_df = app_store_df.copy()
            app_store_df['Store'] = 'App Store'
        
        # Check if App Store DataFrame is empty
        if app_store_df.empty:
            self.logger.info("App Store DataFrame is empty - using Google Play data only")
            unified_df = google_play_df.copy()
        else:
            # Align columns between datasets
            unified_df = self.align_and_merge(google_play_df, app_store_df)
        
        self.logger.info(f"Merged dataset created with {len(unified_df)} total apps")
        return unified_df
    
    def align_and_merge(self, google_play_df: pd.DataFrame, app_store_df: pd.DataFrame) -> pd.DataFrame:
        """Align columns and merge datasets"""
        
        # Get common columns
        gp_columns = set(google_play_df.columns)
        as_columns = set(app_store_df.columns)
        common_columns = gp_columns.intersection(as_columns)
        
        self.logger.info(f"Common columns: {common_columns}")
        
        # Use common columns for both datasets
        if common_columns:
            common_columns_list = list(common_columns)
            gp_aligned = google_play_df[common_columns_list].copy()
            as_aligned = app_store_df[common_columns_list].copy()
        else:
            # Fallback: use all Google Play columns and fill missing ones for App Store
            gp_aligned = google_play_df.copy()
            as_aligned = app_store_df.copy()
            
            # Add missing columns to App Store DataFrame
            for col in gp_aligned.columns:
                if col not in as_aligned.columns:
                    if col in ['Category', 'Genres', 'Content Rating']:
                        as_aligned[col] = 'Unknown'
                    else:
                        as_aligned[col] = 0
        
        # Clean data before merging
        gp_aligned = self.clean_dataframe(gp_aligned)
        as_aligned = self.clean_dataframe(as_aligned)
        
        # Merge datasets
        unified_df = pd.concat([gp_aligned, as_aligned], ignore_index=True, sort=False)
        
        return unified_df
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for consistency"""
        
        df_clean = df.copy()
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['Rating', 'Reviews', 'Size', 'Installs', 'Price']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Ensure text columns are strings
        text_columns = ['App', 'Category', 'Genres', 'Content Rating', 'Store']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).fillna('Unknown')
        
        return df_clean
    
    def save_unified_dataset(self, unified_df: pd.DataFrame, output_dir: Optional[str] = None):
        """Save unified dataset - SIMPLE PATH FIX"""
        
        try:
            # FIXED: Use hardcoded correct paths
            base_dir = Path("data/unified")
            json_path = base_dir / "combined_dataset.json"
            csv_path = base_dir / "combined_dataset.csv"
            
            # Create directory
            base_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving to directory: {base_dir}")
            self.logger.info(f"JSON file: {json_path}")
            self.logger.info(f"CSV file: {csv_path}")
            
            # Remove any existing folders with same names
            for file_path in [json_path, csv_path]:
                if file_path.exists() and file_path.is_dir():
                    self.logger.warning(f"Removing folder: {file_path}")
                    shutil.rmtree(file_path)
            
            # Prepare data for JSON
            self.logger.info("Preparing data for JSON...")
            
            # Create metadata
            metadata = {
                'total_apps': int(len(unified_df)),
                'generated_at': datetime.now().isoformat(),
                'columns': list(unified_df.columns)
            }
            
            # Add store information
            if 'Store' in unified_df.columns:
                store_counts = unified_df['Store'].value_counts()
                metadata['stores'] = {str(k): int(v) for k, v in store_counts.items()}
            
            # Add category information
            if 'Category' in unified_df.columns:
                metadata['categories'] = int(unified_df['Category'].nunique())
                metadata['avg_rating'] = float(unified_df['Rating'].mean()) if 'Rating' in unified_df.columns else 0
                metadata['total_reviews'] = int(unified_df['Reviews'].sum()) if 'Reviews' in unified_df.columns else 0
            
            # Convert DataFrame to JSON-serializable records
            data_records = []
            for i, (_, row) in enumerate(unified_df.iterrows()):
                record = {}
                for col, value in row.items():
                    if pd.isna(value):
                        record[col] = None
                    elif isinstance(value, (np.integer, int)):
                        record[col] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        record[col] = float(value) if not np.isnan(float(value)) else 0
                    else:
                        record[col] = str(value)
                
                data_records.append(record)
                
                # Progress logging
                if i % 1000 == 0 and i > 0:
                    self.logger.info(f"Processed {i:,}/{len(unified_df):,} records...")
            
            unified_data = {
                'metadata': metadata,
                'data': data_records
            }
            
            # Save JSON file
            self.logger.info(f"Saving JSON to: {json_path}")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(unified_data, f, indent=2, ensure_ascii=False)
            
            # Save CSV file
            self.logger.info(f"Saving CSV to: {csv_path}")
            unified_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Verify files were created
            json_success = json_path.exists() and json_path.is_file()
            csv_success = csv_path.exists() and csv_path.is_file()
            
            if json_success and csv_success:
                self.logger.info("=" * 50)
                self.logger.info("✅ FILES SAVED SUCCESSFULLY")
                self.logger.info("=" * 50)
                self.logger.info(f"JSON File: {json_path}")
                self.logger.info(f"   Size: {json_path.stat().st_size:,} bytes")
                self.logger.info(f"CSV File: {csv_path}")  
                self.logger.info(f"   Size: {csv_path.stat().st_size:,} bytes")
                self.logger.info(f"📊 Dataset Summary:")
                self.logger.info(f"   Total apps: {len(unified_df):,}")
                
                if 'Store' in unified_df.columns:
                    for store, count in unified_df['Store'].value_counts().items():
                        self.logger.info(f"   - {store}: {count:,} apps")
                
                if 'Category' in unified_df.columns:
                    self.logger.info(f"   Categories: {unified_df['Category'].nunique()}")
                
                if 'Rating' in unified_df.columns:
                    self.logger.info(f"   Avg Rating: {unified_df['Rating'].mean():.2f}")
                
                self.logger.info("=" * 50)
            else:
                if not json_success:
                    self.logger.error(f"❌ JSON file not created: {json_path}")
                if not csv_success:
                    self.logger.error(f"❌ CSV file not created: {csv_path}")
            
            return json_path
            
        except Exception as e:
            self.logger.error(f"Error in save_unified_dataset: {e}")
            raise
