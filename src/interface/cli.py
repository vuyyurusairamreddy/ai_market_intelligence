import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import Config

class MarketIntelligenceCLI:
    def __init__(self):
        self.config = Config()
        self.df = None
        self.insights = None
        self.load_data()
    
    def load_data(self):
        """Load processed data and insights"""
        try:
            # Load unified dataset
            unified_path = Path(self.config.UNIFIED_DATA_PATH) / self.config.UNIFIED_DATA_FILE
            if unified_path.exists():
                with open(unified_path, 'r') as f:
                    unified_data = json.load(f)
                self.df = pd.DataFrame(unified_data['data'])
                print(f" Loaded {len(self.df)} apps from unified dataset")
            else:
                print(" Unified dataset not found. Please run the main pipeline first.")
                return False
            
            # Load insights
            insights_path = Path(self.config.OUTPUTS_PATH) / self.config.INSIGHTS_FILE
            if insights_path.exists():
                with open(insights_path, 'r') as f:
                    insights_data = json.load(f)
                self.insights = insights_data.get('insights', [])
                print(f" Loaded {len(self.insights)} insights")
            else:
                print(" Insights not found. Some features may be limited.")
                self.insights = []
            
            return True
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return False
    
    def query_top_categories(self, metric: str = 'rating', limit: int = 10):
        """Query top categories by specified metric"""
        if self.df is None:
            print(" No data loaded")
            return
        
        print(f"\n Top {limit} Categories by {metric.title()}")
        print("=" * 50)
        
        if metric.lower() == 'rating':
            top_categories = self.df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(limit)
            for category, rating in top_categories.items():
                app_count = len(self.df[self.df['Category'] == category])
                print(f"{category:25} | Rating: {rating:.2f} | Apps: {app_count:,}")
        
        elif metric.lower() == 'reviews':
            top_categories = self.df.groupby('Category')['Reviews'].sum().sort_values(ascending=False).head(limit)
            for category, reviews in top_categories.items():
                app_count = len(self.df[self.df['Category'] == category])
                print(f"{category:25} | Reviews: {reviews:,.0f} | Apps: {app_count:,}")
        
        elif metric.lower() == 'count':
            top_categories = self.df['Category'].value_counts().head(limit)
            for category, count in top_categories.items():
                avg_rating = self.df[self.df['Category'] == category]['Rating'].mean()
                print(f"{category:25} | Apps: {count:,} | Avg Rating: {avg_rating:.2f}")
        
        else:
            print(f" Unknown metric: {metric}")
    
    def query_top_apps(self, category: str = None, metric: str = 'rating', limit: int = 10):
        """Query top apps by specified criteria"""
        if self.df is None:
            print(" No data loaded")
            return
        
        df_filtered = self.df
        
        if category:
            df_filtered = df_filtered[df_filtered['Category'].str.contains(category, case=False, na=False)]
            if df_filtered.empty:
                print(f" No apps found for category: {category}")
                return
            print(f"\n Top {limit} Apps in {category} by {metric.title()}")
        else:
            print(f"\n Top {limit} Apps Overall by {metric.title()}")
        
        print("=" * 80)
        
        if metric.lower() == 'rating':
            top_apps = df_filtered.nlargest(limit, 'Rating')
        elif metric.lower() == 'reviews':
            top_apps = df_filtered.nlargest(limit, 'Reviews')
        elif metric.lower() == 'score' and 'Cross_Platform_Score' in df_filtered.columns:
            top_apps = df_filtered.nlargest(limit, 'Cross_Platform_Score')
        else:
            print(f" Unknown metric: {metric}")
            return
        
        for idx, app in top_apps.iterrows():
            store = app.get('Store', 'Unknown')
            if metric.lower() == 'rating':
                print(f"{app['App'][:35]:35} | {app['Category'][:15]:15} | Rating: {app['Rating']:.2f} | Reviews: {app['Reviews']:,.0f} | {store}")
            elif metric.lower() == 'reviews':
                print(f"{app['App'][:35]:35} | {app['Category'][:15]:15} | Reviews: {app['Reviews']:,.0f} | Rating: {app['Rating']:.2f} | {store}")
            elif metric.lower() == 'score':
                score = app.get('Cross_Platform_Score', 0)
                print(f"{app['App'][:35]:35} | {app['Category'][:15]:15} | Score: {score:.1f} | Rating: {app['Rating']:.2f} | {store}")
    
    def query_store_comparison(self):
        """Compare performance between stores"""
        if self.df is None or 'Store' not in self.df.columns:
            print(" Store data not available")
            return
        
        print("\n Store Performance Comparison")
        print("=" * 50)
        
        store_stats = self.df.groupby('Store').agg({
            'App': 'count',
            'Rating': 'mean',
            'Reviews': ['mean', 'sum']
        }).round(2)
        
        store_stats.columns = ['App_Count', 'Avg_Rating', 'Avg_Reviews', 'Total_Reviews']
        
        for store, stats in store_stats.iterrows():
            print(f"\n{store}:")
            print(f"  Apps: {stats['App_Count']:,}")
            print(f"  Avg Rating: {stats['Avg_Rating']:.2f}")
            print(f"  Avg Reviews: {stats['Avg_Reviews']:,.0f}")
            print(f"  Total Reviews: {stats['Total_Reviews']:,.0f}")
    
    def query_growth_opportunities(self, limit: int = 10):
        """Find apps with growth opportunities"""
        if self.df is None:
            print(" No data loaded")
            return
        
        print(f"\n Top {limit} Growth Opportunities")
        print("=" * 80)
        
        # High engagement but low ratings
        opportunities = self.df[
            (self.df['Reviews'] > self.df['Reviews'].quantile(0.7)) & 
            (self.df['Rating'] < self.df['Rating'].quantile(0.4))
        ].sort_values('Reviews', ascending=False).head(limit)
        
        if opportunities.empty:
            print(" No clear growth opportunities found with current criteria")
            return
        
        print("Apps with high user engagement but low satisfaction scores:")
        print("(These apps have potential for improvement)")
        print()
        
        for idx, app in opportunities.iterrows():
            store = app.get('Store', 'Unknown')
            print(f"{app['App'][:35]:35} | {app['Category'][:15]:15} | Rating: {app['Rating']:.2f} | Reviews: {app['Reviews']:,.0f} | {store}")
    
    def query_insights(self, insight_type: str = None):
        """Display AI-generated insights"""
        if not self.insights:
            print(" No insights available")
            return
        
        if insight_type:
            filtered_insights = [i for i in self.insights if i.get('type') == insight_type]
            if not filtered_insights:
                print(f" No insights found for type: {insight_type}")
                available_types = list(set(i.get('type', 'unknown') for i in self.insights))
                print(f"Available types: {', '.join(available_types)}")
                return
            insights_to_show = filtered_insights
        else:
            insights_to_show = self.insights
        
        for insight in insights_to_show:
            insight_type = insight.get('type', 'Unknown').replace('_', ' ').title()
            print(f"\n {insight_type}")
            print("=" * 60)
            
            content = insight.get('content', 'No content available')
            print(content)
            
            if 'confidence_metrics' in insight:
                confidence = insight['confidence_metrics']
                conf_level = confidence.get('confidence_level', 'Unknown')
                conf_score = confidence.get('overall_confidence', 0)
                
                # Color coding for confidence levels
                if conf_level == 'High':
                    print(f"\n Confidence: {conf_level} ({conf_score:.2f})")
                elif conf_level == 'Medium':
                    print(f"\n Confidence: {conf_level} ({conf_score:.2f})")
                else:
                    print(f"\n Confidence: {conf_level} ({conf_score:.2f})")
            
            if 'recommendations' in insight:
                print("\n Recommendations:")
                for i, rec in enumerate(insight['recommendations'], 1):
                    print(f"  {i}. {rec}")
            
            print("\n" + "-" * 60)
    
    def query_category_analysis(self, category: str):
        """Analyze specific category in detail"""
        if self.df is None:
            print(" No data loaded")
            return
        
        # Find matching categories (case insensitive, partial match)
        matching_categories = self.df[self.df['Category'].str.contains(category, case=False, na=False)]['Category'].unique()
        
        if len(matching_categories) == 0:
            print(f" No categories found matching: {category}")
            available_categories = sorted(self.df['Category'].unique())
            print(f"Available categories: {', '.join(available_categories[:10])}{'...' if len(available_categories) > 10 else ''}")
            return
        
        if len(matching_categories) > 1:
            print(f"Multiple categories found: {', '.join(matching_categories)}")
            category_name = matching_categories[0]
            print(f"Analyzing: {category_name}")
        else:
            category_name = matching_categories[0]
        
        category_data = self.df[self.df['Category'] == category_name]
        
        print(f"\n Category Analysis: {category_name}")
        print("=" * 60)
        
        # Basic stats
        print(f"Total Apps: {len(category_data):,}")
        print(f"Average Rating: {category_data['Rating'].mean():.2f}")
        print(f"Average Reviews: {category_data['Reviews'].mean():,.0f}")
        print(f"Total Reviews: {category_data['Reviews'].sum():,.0f}")
        
        if 'Store' in category_data.columns:
            store_dist = category_data['Store'].value_counts()
            print(f"Store Distribution: {dict(store_dist)}")
        
        # Top apps in category
        print(f"\n Top 5 Apps in {category_name}:")
        top_apps = category_data.nlargest(5, 'Rating')
        for idx, app in top_apps.iterrows():
            store = app.get('Store', 'Unknown')
            print(f"  • {app['App'][:40]:40} | Rating: {app['Rating']:.2f} | Reviews: {app['Reviews']:,.0f} | {store}")
        
        # Performance insights
        overall_avg_rating = self.df['Rating'].mean()
        overall_avg_reviews = self.df['Reviews'].mean()
        
        print(f"\n Performance vs Market Average:")
        rating_diff = category_data['Rating'].mean() - overall_avg_rating
        reviews_diff = category_data['Reviews'].mean() - overall_avg_reviews
        
        print(f"  Rating: {'+' if rating_diff > 0 else ''}{rating_diff:.2f} vs market average")
        print(f"  Reviews: {'+' if reviews_diff > 0 else ''}{reviews_diff:,.0f} vs market average")
    
    def interactive_mode(self):
        """Start interactive CLI mode"""
        print("\n AI-Powered Market Intelligence CLI")
        print("=" * 50)
        print("Available commands:")
        print("  categories [metric] [limit] - Top categories")
        print("  apps [category] [metric] [limit] - Top apps")
        print("  stores - Store comparison")
        print("  growth [limit] - Growth opportunities")
        print("  insights [type] - AI insights")
        print("  analyze [category] - Category analysis")
        print("  help - Show this help")
        print("  exit - Exit CLI")
        print("\nExamples:")
        print("  categories rating 5")
        print("  apps SOCIAL rating 10")
        print("  insights category_trends")
        print("  analyze games")
        
        while True:
            try:
                command = input("\n> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'exit':
                    print(" Goodbye!")
                    break
                
                elif cmd == 'help':
                    print("\nAvailable commands:")
                    print("  categories [metric] [limit] - Top categories by rating/reviews/count")
                    print("  apps [category] [metric] [limit] - Top apps (optionally filtered by category)")
                    print("  stores - Compare store performance")
                    print("  growth [limit] - Find growth opportunities")
                    print("  insights [type] - Show AI insights (category_trends, growth_opportunities, store_comparison, market_intelligence)")
                    print("  analyze [category] - Detailed category analysis")
                
                elif cmd == 'categories':
                    metric = command[1] if len(command) > 1 else 'rating'
                    limit = int(command[2]) if len(command) > 2 else 10
                    self.query_top_categories(metric, limit)
                
                elif cmd == 'apps':
                    category = command[1] if len(command) > 1 else None
                    metric = command[2] if len(command) > 2 else 'rating'
                    limit = int(command[3]) if len(command) > 3 else 10
                    self.query_top_apps(category, metric, limit)
                
                elif cmd == 'stores':
                    self.query_store_comparison()
                
                elif cmd == 'growth':
                    limit = int(command[1]) if len(command) > 1 else 10
                    self.query_growth_opportunities(limit)
                
                elif cmd == 'insights':
                    insight_type = command[1] if len(command) > 1 else None
                    self.query_insights(insight_type)
                
                elif cmd == 'analyze':
                    if len(command) > 1:
                        category = ' '.join(command[1:])
                        self.query_category_analysis(category)
                    else:
                        print(" Please specify a category to analyze")
                
                else:
                    print(f" Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Market Intelligence CLI')
    parser.add_argument('--interactive', '-i', action='store_true', help='Start interactive mode')
    parser.add_argument('--categories', '-c', type=str, help='Query top categories by metric (rating/reviews/count)')
    parser.add_argument('--apps', '-a', type=str, help='Query top apps, optionally by category')
    parser.add_argument('--stores', '-s', action='store_true', help='Compare store performance')
    parser.add_argument('--growth', '-g', action='store_true', help='Find growth opportunities')
    parser.add_argument('--insights', type=str, help='Show AI insights by type')
    parser.add_argument('--analyze', type=str, help='Analyze specific category')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit number of results')
    
    args = parser.parse_args()
    
    cli = MarketIntelligenceCLI()
    
    if not cli.df is not None:
        print("Failed to load data. Please run the main pipeline first: python main.py")
        return
    
    if args.interactive:
        cli.interactive_mode()
    elif args.categories:
        cli.query_top_categories(args.categories, args.limit)
    elif args.apps:
        cli.query_top_apps(args.apps, 'rating', args.limit)
    elif args.stores:
        cli.query_store_comparison()
    elif args.growth:
        cli.query_growth_opportunities(args.limit)
    elif args.insights:
        cli.query_insights(args.insights)
    elif args.analyze:
        cli.query_category_analysis(args.analyze)
    else:
        print("Use --interactive for interactive mode or --help for available options")

if __name__ == '__main__':
    main()
