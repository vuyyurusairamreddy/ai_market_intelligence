import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

def import_modules():
    """Import all required modules with error handling"""
    try:
        from config.settings import Config
        from src.data_pipeline.google_play_cleaner import GooglePlayStoreCleaner
        from src.data_pipeline.app_store_scraper import AppStoreScraper
        from src.data_pipeline.data_merger import DataMerger
        from src.insights.llm_insights import LLMInsightsEngine
        from src.insights.confidence_scorer import ConfidenceScorer
        from src.reports.report_generator import ReportGenerator
        
        return {
            'Config': Config,
            'GooglePlayStoreCleaner': GooglePlayStoreCleaner,
            'AppStoreScraper': AppStoreScraper,
            'DataMerger': DataMerger,
            'LLMInsightsEngine': LLMInsightsEngine,
            'ConfidenceScorer': ConfidenceScorer,
            'ReportGenerator': ReportGenerator
        }
    except ImportError as e:
        logging.error(f"Import error: {e}")
        logging.error("Make sure all source files are in the correct directory structure")
        return None

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/unified",
        "outputs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logging.info("Created necessary directories")

def phase_1_google_play_pipeline(modules):
    """Phase 1: Process Google Play Store data"""
    logging.info("Starting Phase 1: Google Play Store Data Pipeline")
    
    config = modules['Config']()
    cleaner = modules['GooglePlayStoreCleaner'](config.RAW_DATA_PATH)
    
    try:
        # Process data
        cleaned_df = cleaner.process()
        
        # Save cleaned data
        output_path = Path(config.PROCESSED_DATA_PATH) / config.CLEANED_DATA_FILE
        cleaned_df.to_csv(output_path, index=False)
        
        logging.info(f"Phase 1 completed. Cleaned dataset saved with {len(cleaned_df)} apps")
        return cleaned_df
        
    except Exception as e:
        logging.error(f"Phase 1 failed: {e}")
        return None

def phase_2_app_store_pipeline(modules, categories_to_scrape=None):
    """Phase 2: Scrape App Store data and merge"""
    logging.info("Starting Phase 2: App Store Scraping and Data Merger")
    
    config = modules['Config']()
    
    # Default categories if none provided
    if categories_to_scrape is None:
        categories_to_scrape = [
            'SOCIAL', 'ENTERTAINMENT', 'PHOTOGRAPHY', 'PRODUCTIVITY',
            'GAME', 'FINANCE', 'HEALTH_AND_FITNESS', 'SHOPPING'
        ]
    
    try:
        # Load cleaned Google Play data
        cleaned_data_path = Path(config.PROCESSED_DATA_PATH) / config.CLEANED_DATA_FILE
        if not cleaned_data_path.exists():
            logging.error("Cleaned Google Play data not found. Run Phase 1 first.")
            return None
        
        google_play_df = pd.read_csv(cleaned_data_path)
        
        # Check if API keys are available
        if not config.RAPIDAPI_KEY:
            logging.warning("RapidAPI key not found. Skipping App Store scraping.")
            return google_play_df
        
        # Scrape App Store data
        scraper = modules['AppStoreScraper']()
        app_store_data = scraper.scrape_competitive_apps(categories_to_scrape, apps_per_category=15)
        
        # Merge datasets
        merger = modules['DataMerger']()
        unified_df = merger.merge_datasets(google_play_df, app_store_data)
        
        # Save unified dataset
        unified_path = Path(config.UNIFIED_DATA_PATH) / config.UNIFIED_DATA_FILE
        merger.save_unified_dataset(unified_df, str(unified_path))
        
        logging.info(f"Phase 2 completed. Unified dataset saved with {len(unified_df)} apps")
        return unified_df
        
    except Exception as e:
        logging.error(f"Phase 2 failed: {e}")
        logging.warning("Continuing with Google Play data only")
        # Return Google Play data only if App Store scraping fails
        try:
            cleaned_data_path = Path(config.PROCESSED_DATA_PATH) / config.CLEANED_DATA_FILE
            return pd.read_csv(cleaned_data_path)
        except:
            return None

def phase_3_insights_generation(modules, df):
    """Phase 3: Generate LLM insights with confidence scoring"""
    logging.info("Starting Phase 3: LLM Insights Generation")
    
    config = modules['Config']()
    
    try:
        # Check if API key is available
        if not config.PERPLEXITY_API_KEY:
            logging.error("Perplexity API key not found. Cannot generate insights.")
            return None
        
        # Generate insights
        insights_engine = modules['LLMInsightsEngine']()
        insights = insights_engine.generate_all_insights(df)
        
        if not insights:
            logging.error("No insights generated")
            return None
        
        # Add confidence scores
        confidence_scorer = modules['ConfidenceScorer']()
        
        scored_insights = []
        for insight in insights:
            scored_insight = confidence_scorer.calculate_insight_confidence(insight, df)
            
            # Add recommendations
            recommendations = confidence_scorer.generate_recommendations(scored_insight)
            scored_insight['recommendations'] = recommendations
            
            scored_insights.append(scored_insight)
        
        # Save insights
        insights_path = Path(config.OUTPUTS_PATH) / config.INSIGHTS_FILE
        
        insights_output = {
            'metadata': {
                'total_insights': len(scored_insights),
                'generated_at': pd.Timestamp.now().isoformat(),
                'data_summary': {
                    'total_apps': len(df),
                    'categories': df['Category'].nunique(),
                    'stores': df['Store'].nunique() if 'Store' in df.columns else 1
                }
            },
            'insights': scored_insights
        }
        
        with open(insights_path, 'w') as f:
            json.dump(insights_output, f, indent=2, default=str)
        
        logging.info(f"Phase 3 completed. Generated {len(scored_insights)} insights")
        return scored_insights
        
    except Exception as e:
        logging.error(f"Phase 3 failed: {e}")
        return None

def phase_4_report_generation(modules, df, insights):
    """Phase 4: Generate executive report"""
    logging.info("Starting Phase 4: Report Generation")
    
    config = modules['Config']()
    
    try:
        report_generator = modules['ReportGenerator']()
        
        # Generate report
        report_path = Path(config.OUTPUTS_PATH) / config.REPORT_FILE
        
        # Handle case where insights is None
        if insights is None:
            insights = []
            logging.warning("No insights available for report generation")
        
        report_generator.generate_executive_report(df, insights, str(report_path))
        
        logging.info(f"Phase 4 completed. Executive report saved to {report_path}")
        return True
        
    except Exception as e:
        logging.error(f"Phase 4 failed: {e}")
        return False

def phase_5_d2c_extension(modules):
    """Phase 5: D2C Dataset Extension with funnel insights and creative generation"""
    logging.info("Starting Phase 5: D2C Dataset Extension")
    
    config = modules['Config']()
    
    try:
        # Import D2C modules dynamically
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from src.data_pipeline.d2c_processor import D2CProcessor
        from src.insights.d2c_creative_generator import D2CCreativeGenerator
        
        # Check if API key is available
        if not config.PERPLEXITY_API_KEY:
            logging.error("Perplexity API key not found. Cannot generate creative content.")
            return None, None
        
        # Process D2C data
        d2c_processor = D2CProcessor(config.RAW_DATA_PATH)
        d2c_insights = d2c_processor.process_d2c_data()
        
        # Generate creative content
        creative_generator = D2CCreativeGenerator()
        creative_outputs = creative_generator.generate_all_creative_content(d2c_insights)
        
        # Save D2C insights and creative outputs
        d2c_output = {
            'metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'total_campaigns': d2c_insights['data_summary']['total_campaigns'],
                'total_channels': d2c_insights['data_summary']['channels'],
                'creative_outputs_generated': len(creative_outputs)
            },
            'funnel_insights': {
                'data_summary': d2c_insights['data_summary'],
                'channel_analysis': d2c_insights['channel_analysis'],
                'seo_analysis': d2c_insights['seo_analysis'],
                'growth_patterns': d2c_insights['growth_patterns'],
                'recommendations': d2c_insights['recommendations']
            },
            'creative_outputs': creative_outputs
        }
        
        # Save D2C results
        d2c_output_path = Path(config.OUTPUTS_PATH) / "d2c_insights_and_creative.json"
        with open(d2c_output_path, 'w') as f:
            json.dump(d2c_output, f, indent=2, default=str)
        
        # Save processed D2C data
        d2c_data_path = Path(config.PROCESSED_DATA_PATH) / "d2c_processed_data.csv"
        d2c_insights['processed_data'].to_csv(d2c_data_path, index=False)
        
        logging.info(f"Phase 5 completed. Generated {len(creative_outputs)} creative outputs and comprehensive D2C analysis")
        return d2c_insights, creative_outputs
        
    except FileNotFoundError as e:
        logging.error(f"D2C dataset file not found: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Phase 5 failed: {e}")
        return None, None

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    
    # Check if .env file exists
    env_path = Path('.env')
    if not env_path.exists():
        issues.append("  .env file not found. Create one with your API keys.")
    
    # Check data directory
    data_dir = Path('data/raw')
    if not data_dir.exists():
        issues.append("  data/raw directory not found.")
    
    required_files = [
        'data/raw/googleplaystore.csv',
        'data/raw/googleplaystore_user_reviews.csv'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f" Required file missing: {file_path}")
    
    return issues

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print(" AI-POWERED MARKET INTELLIGENCE SYSTEM")
    print("="*70)
    
    # Check requirements first
    print("\n Checking Requirements...")
    issues = check_requirements()
    
    if issues:
        print(" Issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\n Please fix the issues above before running the system.")
        return
    
    print(" All requirements met")
    
    # Import modules
    print("\n Loading Modules...")
    modules = import_modules()
    if modules is None:
        print(" Failed to import required modules. Check file structure.")
        return
    
    print(" All modules loaded successfully")
    
    # Create directories
    create_directories()
    
    # Get configuration
    config = modules['Config']()
    
    # Check data files
    google_play_path = Path(config.RAW_DATA_PATH) / config.GOOGLE_PLAY_FILE
    google_play_reviews_path = Path(config.RAW_DATA_PATH) / config.GOOGLE_PLAY_REVIEWS_FILE
    d2c_path = Path(config.RAW_DATA_PATH) / "D2C_Synthetic_Dataset.csv"
    
    print("\n Checking Data Files...")
    print(f"   Google Play Store: {'✅' if google_play_path.exists() else '❌'}")
    print(f"   Google Play Reviews: {'✅' if google_play_reviews_path.exists() else '❌'}")
    print(f"   D2C Dataset: {'✅' if d2c_path.exists() else '  (Optional)'}")
    
    if not google_play_path.exists() or not google_play_reviews_path.exists():
        print("\n Required data files missing:")
        print(f"   - {google_play_path}")
        print(f"   - {google_play_reviews_path}")
        print("\n Please place the required CSV files in the data/raw/ directory")
        return
    
    # Execute phases
    print("\n" + "="*70)
    print(" PIPELINE EXECUTION")
    print("="*70)
    
    # Phase 1: Google Play Store Data Pipeline
    print("\n[PHASE 1]  Processing Google Play Store Data...")
    cleaned_df = phase_1_google_play_pipeline(modules)
    
    if cleaned_df is None:
        print(" Phase 1 failed. Cannot continue.")
        return
    
    print(f" Phase 1 completed: {len(cleaned_df):,} apps processed")
    
    # Phase 2: App Store Scraping and Data Merger  
    print("\n[PHASE 2]  Scraping App Store Data and Merging...")
    unified_df = phase_2_app_store_pipeline(modules)
    
    if unified_df is None:
        print(" Phase 2 failed. Cannot continue.")
        return
    
    stores = unified_df['Store'].nunique() if 'Store' in unified_df.columns else 1
    print(f" Phase 2 completed: {len(unified_df):,} total apps from {stores} store(s)")
    
    # Phase 3: LLM Insights Generation
    print("\n[PHASE 3]  Generating AI Insights...")
    insights = phase_3_insights_generation(modules, unified_df)
    
    if insights is None:
        print("  Phase 3 completed with limited functionality (no API key or errors)")
        insights = []
    else:
        print(f" Phase 3 completed: {len(insights)} AI insights generated")
    
    # Phase 4: Report Generation
    print("\n[PHASE 4]  Generating Executive Report...")
    report_success = phase_4_report_generation(modules, unified_df, insights)
    
    if report_success:
        print(" Phase 4 completed: Executive report generated")
    else:
        print("  Phase 4 completed with issues")
    
    # Phase 5: D2C Extension (Bonus)
    if d2c_path.exists():
        print("\n[PHASE 5]  Processing D2C Dataset and Generating Creative Content...")
        d2c_insights, creative_outputs = phase_5_d2c_extension(modules)
        
        if d2c_insights and creative_outputs:
            print(f"Phase 5 completed: D2C analysis with {len(creative_outputs)} creative outputs")
        else:
            print("Phase 5 completed with limited functionality")
    else:
        print(f"\n[PHASE 5] Skipped: D2C dataset not found")
        print("   Place D2C_Synthetic_Dataset.csv in data/raw/ to enable Phase 5")
    
    # Summary
    print("\n" + "="*70)
    print(" EXECUTION SUMMARY")
    print("="*70)
    
    print(f" Total apps processed: {len(unified_df):,}")
    
    if 'Store' in unified_df.columns:
        store_counts = unified_df['Store'].value_counts()
        for store, count in store_counts.items():
            print(f"   • {store}: {count:,} apps")
    
    print(f" Categories analyzed: {unified_df['Category'].nunique()}")
    print(f" AI insights generated: {len(insights) if insights else 0}")
    
    if d2c_path.exists():
        print(" D2C analysis:  Available")
        print(" Creative outputs:  Generated")
    
    print(f"\n Generated Files:")
    print(f"   • Cleaned dataset: data/processed/{config.CLEANED_DATA_FILE}")
    print(f"   • Unified dataset: data/unified/{config.UNIFIED_DATA_FILE}")
    print(f"   • AI insights: outputs/{config.INSIGHTS_FILE}")
    print(f"   • Executive report: outputs/{config.REPORT_FILE}")
    
    if d2c_path.exists():
        print(f"   • D2C analysis: outputs/d2c_insights_and_creative.json")
        print(f"   • D2C processed data: data/processed/d2c_processed_data.csv")
    
    print(f"\n Next Steps:")
    print("   1. Launch interactive dashboard:")
    print("      streamlit run src/interface/streamlit_app.py")
    print("   2. Use CLI for quick queries:")
    print("      python src/interface/cli.py --interactive")
    print("   3. Check generated reports in outputs/ directory")
    
    print(f"\n System ready for comprehensive market intelligence analysis!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {e}")
        print(f"\n Unexpected error: {e}")
        print(" Check the logs for more details")
