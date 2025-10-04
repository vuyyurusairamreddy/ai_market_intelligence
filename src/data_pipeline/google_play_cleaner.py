import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict
import logging

class GooglePlayStoreCleaner:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Google Play Store data and reviews with proper encoding"""
        try:
            # Try UTF-8 first, then fallback to other encodings
            apps_df = self.safe_csv_read(f"{self.data_path}googleplaystore.csv")
            reviews_df = self.safe_csv_read(f"{self.data_path}googleplaystore_user_reviews.csv")
            
            self.logger.info(f"Loaded {len(apps_df)} apps and {len(reviews_df)} reviews")
            return apps_df, reviews_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def safe_csv_read(self, file_path: str) -> pd.DataFrame:
        """Safely read CSV with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
        
        for encoding in encodings:
            try:
                self.logger.info(f"Attempting to read {file_path} with {encoding} encoding...")
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"✅ Successfully read with {encoding} encoding")
                return df
                
            except UnicodeDecodeError as e:
                self.logger.warning(f"❌ {encoding} encoding failed: {e}")
                continue
            except Exception as e:
                self.logger.warning(f"❌ Error with {encoding}: {e}")
                continue
        
        # Final attempt with error handling
        try:
            self.logger.info("Final attempt with utf-8 and error handling...")
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            self.logger.warning("⚠️ Read with character replacement - some characters may be corrupted")
            return df
        except Exception as e:
            raise Exception(f"Could not read {file_path} with any encoding method: {e}")
    
    def clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text data to handle encoding issues"""
        text_columns = ['App', 'Category', 'Genres', 'Content Rating']
        
        for col in text_columns:
            if col in df.columns:
                # Replace common encoding issues
                df[col] = df[col].astype(str)
                df[col] = df[col].replace('â€™', "'")  # Fix apostrophes
                df[col] = df[col].replace('â€œ', '"')  # Fix quotes
                df[col] = df[col].replace('â€', '"')   # Fix quotes
                df[col] = df[col].replace('Ã©', 'é')   # Fix accents
                df[col] = df[col].replace('Ã¡', 'á')   # Fix accents
                df[col] = df[col].replace('Ã³', 'ó')   # Fix accents
                
                # Remove any remaining non-printable characters
                df[col] = df[col].apply(self.clean_string)
        
        return df
    
    def clean_string(self, text):
        """Clean individual strings of problematic characters"""
        if pd.isna(text):
            return text
        
        text = str(text)
        
        # Remove or replace problematic characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text if text else 'Unknown'
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        initial_count = len(df)
        df_clean = df.copy()
        
        self.logger.info(f"Starting validation with {initial_count} records")
        
        # Clean text data first
        df_clean = self.clean_text_data(df_clean)
        
        # 1. Remove rows where Category is numeric (invalid category)
        def is_numeric_category(cat):
            if pd.isna(cat):
                return False
            try:
                float(str(cat))
                return True
            except ValueError:
                return False
        
        numeric_categories = df_clean['Category'].apply(is_numeric_category)
        invalid_category_count = numeric_categories.sum()
        
        if invalid_category_count > 0:
            self.logger.warning(f"Found {invalid_category_count} rows with numeric categories - removing")
            df_clean = df_clean[~numeric_categories]
        
        # 2. Fix Rating column - remove values > 5 or < 0
        df_clean['Rating'] = pd.to_numeric(df_clean['Rating'], errors='coerce')
        invalid_ratings = (df_clean['Rating'] > 5.0) | (df_clean['Rating'] < 0.0)
        invalid_rating_count = invalid_ratings.sum()
        
        if invalid_rating_count > 0:
            self.logger.warning(f"Found {invalid_rating_count} rows with invalid ratings (>5 or <0) - removing")
            df_clean = df_clean[~invalid_ratings]
        
        # 3. Remove rows with 0 reviews but non-zero rating (suspicious data)
        df_clean['Reviews'] = pd.to_numeric(df_clean['Reviews'], errors='coerce')
        suspicious_reviews = (df_clean['Reviews'] == 0) & (df_clean['Rating'] > 0)
        suspicious_count = suspicious_reviews.sum()
        
        if suspicious_count > 0:
            self.logger.warning(f"Found {suspicious_count} rows with 0 reviews but rating > 0 - removing")
            df_clean = df_clean[~suspicious_reviews]
        
        # 4. Validate App names (should not be purely numeric)
        def is_valid_app_name(name):
            if pd.isna(name) or str(name).strip() == '':
                return False
            # Check if app name is purely numeric
            try:
                float(str(name))
                return False  # Purely numeric app names are suspicious
            except ValueError:
                return True
        
        valid_app_names = df_clean['App'].apply(is_valid_app_name)
        invalid_app_count = (~valid_app_names).sum()
        
        if invalid_app_count > 0:
            self.logger.warning(f"Found {invalid_app_count} rows with invalid app names - removing")
            df_clean = df_clean[valid_app_names]
        
        # 5. Category validation
        valid_categories = {
            'ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY', 'BOOKS_AND_REFERENCE',
            'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'EDUCATION', 'ENTERTAINMENT',
            'EVENTS', 'FINANCE', 'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',
            'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL', 'SOCIAL',
            'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL', 'TOOLS', 'PERSONALIZATION',
            'PRODUCTIVITY', 'PARENTING', 'WEATHER', 'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES',
            'MAPS_AND_NAVIGATION', 'MUSIC_AND_AUDIO'
        }
        
        def is_valid_category(cat):
            if pd.isna(cat):
                return False
            cat_str = str(cat).upper().strip()
            # Check if it's one of the known categories or a reasonable category name
            if cat_str in valid_categories:
                return True
            # Allow categories that are text-based and reasonable length
            if len(cat_str) > 50 or len(cat_str) < 3:
                return False
            return True
        
        valid_category_mask = df_clean['Category'].apply(is_valid_category)
        invalid_category_count_2 = (~valid_category_mask).sum()
        
        if invalid_category_count_2 > 0:
            self.logger.warning(f"Found {invalid_category_count_2} rows with unrecognized categories - removing")
            df_clean = df_clean[valid_category_mask]
        
        # 6. Remove obvious data corruption (where columns are shifted)
        def detect_data_shift(row):
            try:
                # If Category looks like a rating (small decimal number)
                if pd.notna(row['Category']):
                    cat_val = str(row['Category'])
                    if re.match(r'^\d{1,2}\.?\d*$', cat_val) and float(cat_val) <= 5:
                        return True
                
                # If App name looks like a category
                if pd.notna(row['App']):
                    app_val = str(row['App']).upper()
                    if app_val in valid_categories:
                        return True
                
                return False
            except:
                return True  # If error in processing, mark as suspicious
        
        data_shift_mask = df_clean.apply(detect_data_shift, axis=1)
        data_shift_count = data_shift_mask.sum()
        
        if data_shift_count > 0:
            self.logger.warning(f"Found {data_shift_count} rows with suspected data shifts - removing")
            df_clean = df_clean[~data_shift_mask]
        
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        
        self.logger.info(f"Data validation completed: {removed_count} invalid records removed ({removed_count/initial_count*100:.1f}%)")
        self.logger.info(f"Clean dataset has {final_count} records")
        
        return df_clean
    
    def clean_apps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the main apps dataset"""
        # First validate and clean data
        df_clean = self.validate_and_clean_data(df)
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['App'], keep='first')
        duplicate_count = initial_count - len(df_clean)
        if duplicate_count > 0:
            self.logger.info(f"Removed {duplicate_count} duplicate apps")
        
        # Clean Size column
        df_clean['Size'] = self.clean_size_column(df_clean['Size'])
        
        # Clean Installs column
        df_clean['Installs'] = self.clean_installs_column(df_clean['Installs'])
        
        # Clean Price column
        df_clean['Price'] = self.clean_price_column(df_clean['Price'])
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # Clean Last Updated column
        df_clean['Last Updated'] = pd.to_datetime(df_clean['Last Updated'], errors='coerce')
        
        # Add derived features
        df_clean = self.add_derived_features(df_clean)
        
        # Final validation
        df_clean = df_clean[(df_clean['Rating'] >= 1.0) & (df_clean['Rating'] <= 5.0)]
        df_clean = df_clean[df_clean['Reviews'] >= 0]
        
        self.logger.info(f"Final cleaned dataset has {len(df_clean)} apps")
        return df_clean
    
    def clean_size_column(self, size_series: pd.Series) -> pd.Series:
        """Convert size to numeric MB values"""
        def convert_size(size_str):
            if pd.isna(size_str) or size_str == 'Varies with device':
                return np.nan
            
            size_str = str(size_str).strip()
            
            if not re.search(r'\d', size_str):
                return np.nan
            
            try:
                if 'M' in size_str:
                    numbers = re.findall(r'[\d.]+', size_str)
                    if numbers:
                        return float(numbers[0])
                elif 'k' in size_str or 'K' in size_str:
                    numbers = re.findall(r'[\d.]+', size_str)
                    if numbers:
                        return float(numbers[0]) / 1024
                elif 'G' in size_str:
                    numbers = re.findall(r'[\d.]+', size_str)
                    if numbers:
                        return float(numbers[0]) * 1024
                else:
                    numbers = re.findall(r'[\d.]+', size_str)
                    if numbers:
                        return float(numbers[0])
                    return np.nan
            except (ValueError, IndexError):
                return np.nan
            
            return np.nan
        
        return size_series.apply(convert_size)
    
    def clean_installs_column(self, installs_series: pd.Series) -> pd.Series:
        """Convert installs to numeric values"""
        def convert_installs(install_str):
            if pd.isna(install_str):
                return np.nan
            
            install_str = str(install_str).replace(',', '').replace('+', '').replace('Free', '0')
            install_str = re.sub(r'[^\d.]', '', install_str)
            
            try:
                if install_str:
                    return float(install_str)
                return 0.0
            except ValueError:
                return 0.0
        
        return installs_series.apply(convert_installs)
    
    def clean_price_column(self, price_series: pd.Series) -> pd.Series:
        """Convert price to numeric USD values"""
        def convert_price(price_str):
            if pd.isna(price_str) or price_str == '0' or price_str == 'Free':
                return 0.0
            
            price_str = str(price_str).replace('$', '').replace(',', '')
            price_str = re.sub(r'[^\d.]', '', price_str)
            
            try:
                if price_str:
                    return float(price_str)
                return 0.0
            except ValueError:
                return 0.0
        
        return price_series.apply(convert_price)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        # Fill missing ratings with category median
        for category in df['Category'].unique():
            if pd.notna(category):
                category_median = df[df['Category'] == category]['Rating'].median()
                if pd.notna(category_median):
                    df.loc[(df['Category'] == category) & (df['Rating'].isna()), 'Rating'] = category_median
        
        # Fill remaining missing ratings with overall median
        overall_median = df['Rating'].median()
        if pd.notna(overall_median):
            df['Rating'].fillna(overall_median, inplace=True)
        
        # Fill missing reviews with 0
        df['Reviews'].fillna(0, inplace=True)
        
        # Fill missing size with category median
        for category in df['Category'].unique():
            if pd.notna(category):
                category_median = df[df['Category'] == category]['Size'].median()
                if pd.notna(category_median):
                    df.loc[(df['Category'] == category) & (df['Size'].isna()), 'Size'] = category_median
        
        # Drop rows with critical missing values
        initial_count = len(df)
        df = df.dropna(subset=['App', 'Category'])
        dropped_count = initial_count - len(df)
        
        if dropped_count > 0:
            self.logger.info(f"Dropped {dropped_count} rows with missing critical values")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis"""
        # Rating category
        df['Rating_Category'] = pd.cut(df['Rating'], 
                                     bins=[0, 3.0, 4.0, 4.5, 5.0], 
                                     labels=['Poor', 'Average', 'Good', 'Excellent'],
                                     include_lowest=True)
        
        # Install category
        df['Install_Category'] = pd.cut(df['Installs'], 
                                      bins=[0, 1000, 50000, 1000000, float('inf')], 
                                      labels=['Low', 'Medium', 'High', 'Very High'],
                                      include_lowest=True)
        
        # Price category
        df['Price_Category'] = df['Price'].apply(lambda x: 'Free' if x == 0 else 'Paid')
        
        # Days since last update
        current_date = pd.Timestamp.now()
        df['Days_Since_Update'] = (current_date - df['Last Updated']).dt.days
        
        # Quality score
        df['Quality_Score'] = (df['Rating'] * 0.7 + 
                              (np.log1p(df['Reviews']) / np.log1p(df['Reviews']).max()) * 5 * 0.3)
        
        return df
    
    def aggregate_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate review sentiment by app"""
        # Clean sentiment columns
        reviews_df['Sentiment_Polarity'] = pd.to_numeric(reviews_df['Sentiment_Polarity'], errors='coerce')
        reviews_df['Sentiment_Subjectivity'] = pd.to_numeric(reviews_df['Sentiment_Subjectivity'], errors='coerce')
        
        # Aggregate by app
        review_agg = reviews_df.groupby('App').agg({
            'Sentiment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Neutral',
            'Sentiment_Polarity': 'mean',
            'Sentiment_Subjectivity': 'mean',
            'Translated_Review': 'count'
        }).reset_index()
        
        review_agg.columns = ['App', 'Dominant_Sentiment', 'Avg_Sentiment_Polarity', 
                             'Avg_Sentiment_Subjectivity', 'Review_Count']
        
        return review_agg
    
    def process(self) -> pd.DataFrame:
        """Main processing pipeline"""
        # Load data with proper encoding
        apps_df, reviews_df = self.load_data()
        
        # Clean apps data with comprehensive validation
        cleaned_apps = self.clean_apps_data(apps_df)
        
        # Aggregate reviews
        review_summary = self.aggregate_reviews(reviews_df)
        
        # Merge with review data
        final_df = cleaned_apps.merge(review_summary, on='App', how='left')
        
        # Fill missing review data
        final_df['Review_Count'].fillna(0, inplace=True)
        final_df['Dominant_Sentiment'].fillna('Neutral', inplace=True)
        final_df['Avg_Sentiment_Polarity'].fillna(0, inplace=True)
        final_df['Avg_Sentiment_Subjectivity'].fillna(0.5, inplace=True)
        
        # Final quality check
        self.logger.info("Performing final quality checks...")
        
        valid_ratings = (final_df['Rating'] >= 1.0) & (final_df['Rating'] <= 5.0)
        invalid_rating_final = (~valid_ratings).sum()
        
        if invalid_rating_final > 0:
            self.logger.warning(f"Removing {invalid_rating_final} records with invalid final ratings")
            final_df = final_df[valid_ratings]
        
        self.logger.info(f"Processing completed successfully:")
        self.logger.info(f"  - Final dataset: {len(final_df)} apps")
        self.logger.info(f"  - Rating range: {final_df['Rating'].min():.1f} - {final_df['Rating'].max():.1f}")
        self.logger.info(f"  - Categories: {final_df['Category'].nunique()}")
        self.logger.info(f"  - Average rating: {final_df['Rating'].mean():.2f}")
        
        return final_df
