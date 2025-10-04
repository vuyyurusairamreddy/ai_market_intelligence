import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

class D2CProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
    
    def load_d2c_data(self, filename: str = "D2C_Synthetic_Dataset.csv") -> pd.DataFrame:
        """Load D2C synthetic dataset"""
        try:
            file_path = Path(self.data_path) / filename
            df = pd.read_csv(file_path)
            
            self.logger.info(f"Loaded D2C dataset with {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading D2C data: {e}")
            raise
    
    def clean_d2c_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate D2C dataset"""
        df_clean = df.copy()
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'spend_usd', 'impressions', 'clicks', 'installs', 'signups',
            'first_purchase', 'repeat_purchase', 'revenue_usd', 'avg_position',
            'monthly_search_volume', 'conversion_rate'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill missing values with 0 for metrics
        metric_columns = ['spend_usd', 'impressions', 'clicks', 'installs', 'signups',
                         'first_purchase', 'repeat_purchase', 'revenue_usd']
        for col in metric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Remove rows with critical missing data
        df_clean = df_clean.dropna(subset=['campaign_id', 'channel'])
        
        self.logger.info(f"Cleaned D2C dataset: {len(df_clean)} records remaining")
        return df_clean
    
    def calculate_funnel_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive funnel metrics"""
        df_metrics = df.copy()
        
        # Basic conversion rates
        df_metrics['ctr'] = np.where(df_metrics['impressions'] > 0, 
                                   df_metrics['clicks'] / df_metrics['impressions'], 0)
        
        df_metrics['install_rate'] = np.where(df_metrics['clicks'] > 0,
                                            df_metrics['installs'] / df_metrics['clicks'], 0)
        
        df_metrics['signup_rate'] = np.where(df_metrics['installs'] > 0,
                                           df_metrics['signups'] / df_metrics['installs'], 0)
        
        df_metrics['first_purchase_rate'] = np.where(df_metrics['signups'] > 0,
                                                   df_metrics['first_purchase'] / df_metrics['signups'], 0)
        
        df_metrics['repeat_purchase_rate'] = np.where(df_metrics['first_purchase'] > 0,
                                                    df_metrics['repeat_purchase'] / df_metrics['first_purchase'], 0)
        
        # Financial metrics
        df_metrics['cac'] = np.where(df_metrics['signups'] > 0,
                                   df_metrics['spend_usd'] / df_metrics['signups'], 0)
        
        df_metrics['roas'] = np.where(df_metrics['spend_usd'] > 0,
                                    df_metrics['revenue_usd'] / df_metrics['spend_usd'], 0)
        
        df_metrics['cpc'] = np.where(df_metrics['clicks'] > 0,
                                   df_metrics['spend_usd'] / df_metrics['clicks'], 0)
        
        df_metrics['cpi'] = np.where(df_metrics['installs'] > 0,
                                   df_metrics['spend_usd'] / df_metrics['installs'], 0)
        
        df_metrics['ltv_estimate'] = np.where(df_metrics['first_purchase'] > 0,
                                            df_metrics['revenue_usd'] / df_metrics['first_purchase'], 0)
        
        # Retention and engagement metrics
        df_metrics['retention_rate'] = df_metrics['repeat_purchase_rate']
        
        df_metrics['funnel_efficiency'] = (
            df_metrics['ctr'] * 0.2 +
            df_metrics['install_rate'] * 0.25 +
            df_metrics['signup_rate'] * 0.25 +
            df_metrics['first_purchase_rate'] * 0.3
        )
        
        # SEO performance metrics
        df_metrics['seo_opportunity_score'] = np.where(
            (df_metrics['monthly_search_volume'] > 0) & (df_metrics['avg_position'] > 0),
            (df_metrics['monthly_search_volume'] / df_metrics['avg_position']) * df_metrics['conversion_rate'],
            0
        )
        
        self.logger.info("Calculated comprehensive funnel metrics")
        return df_metrics
    
    def analyze_channel_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance by marketing channel"""
        channel_analysis = {}
        
        channel_metrics = df.groupby('channel').agg({
            'spend_usd': 'sum',
            'revenue_usd': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'installs': 'sum',
            'signups': 'sum',
            'first_purchase': 'sum',
            'repeat_purchase': 'sum',
            'cac': 'mean',
            'roas': 'mean',
            'ctr': 'mean',
            'funnel_efficiency': 'mean',
            'campaign_id': 'count'
        }).round(2)
        
        channel_metrics.columns = [
            'Total_Spend', 'Total_Revenue', 'Total_Impressions', 'Total_Clicks',
            'Total_Installs', 'Total_Signups', 'Total_First_Purchases', 
            'Total_Repeat_Purchases', 'Avg_CAC', 'Avg_ROAS', 'Avg_CTR',
            'Avg_Funnel_Efficiency', 'Campaign_Count'
        ]
        
        # Calculate channel-level derived metrics
        channel_metrics['Overall_Conversion_Rate'] = (
            channel_metrics['Total_Signups'] / channel_metrics['Total_Impressions']
        ).fillna(0)
        
        channel_metrics['Channel_ROI'] = (
            (channel_metrics['Total_Revenue'] - channel_metrics['Total_Spend']) / 
            channel_metrics['Total_Spend']
        ).fillna(0)
        
        channel_analysis['performance'] = channel_metrics.to_dict('index')
        
        # Identify best and worst performing channels
        channel_analysis['best_roas'] = channel_metrics['Avg_ROAS'].idxmax()
        channel_analysis['worst_roas'] = channel_metrics['Avg_ROAS'].idxmin()
        channel_analysis['best_efficiency'] = channel_metrics['Avg_Funnel_Efficiency'].idxmax()
        channel_analysis['lowest_cac'] = channel_metrics['Avg_CAC'].idxmin()
        
        return channel_analysis
    
    def analyze_seo_opportunities(self, df: pd.DataFrame) -> Dict:
        """Analyze SEO growth opportunities"""
        seo_analysis = {}
        
        # SEO performance by category
        seo_metrics = df.groupby('seo_category').agg({
            'monthly_search_volume': 'mean',
            'avg_position': 'mean',
            'conversion_rate': 'mean',
            'seo_opportunity_score': 'mean',
            'revenue_usd': 'sum',
            'campaign_id': 'count'
        }).round(2)
        
        seo_metrics.columns = [
            'Avg_Search_Volume', 'Avg_Position', 'Avg_Conversion_Rate',
            'Opportunity_Score', 'Total_Revenue', 'Campaign_Count'
        ]
        
        # Identify high-opportunity categories
        # High search volume + poor position + decent conversion rate = opportunity
        seo_metrics['Growth_Potential'] = (
            (seo_metrics['Avg_Search_Volume'] / seo_metrics['Avg_Search_Volume'].max()) * 0.4 +
            ((10 - seo_metrics['Avg_Position']) / 10) * 0.4 +
            seo_metrics['Avg_Conversion_Rate'] * 0.2
        ) * 100
        
        seo_analysis['category_performance'] = seo_metrics.to_dict('index')
        
        # Top opportunities (high search volume, poor ranking)
        high_volume_poor_ranking = df[
            (df['monthly_search_volume'] > df['monthly_search_volume'].quantile(0.7)) &
            (df['avg_position'] > df['avg_position'].quantile(0.6))
        ].groupby('seo_category').agg({
            'monthly_search_volume': 'mean',
            'avg_position': 'mean',
            'conversion_rate': 'mean'
        }).round(2)
        
        seo_analysis['high_opportunity_categories'] = high_volume_poor_ranking.to_dict('index')
        
        # Best performing SEO categories
        top_seo = seo_metrics.nlargest(5, 'Opportunity_Score')
        seo_analysis['top_performing_categories'] = top_seo.to_dict('index')
        
        return seo_analysis
    
    def identify_growth_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify growth patterns and anomalies"""
        patterns = {}
        
        # High-performing campaigns
        high_performers = df[
            (df['roas'] > df['roas'].quantile(0.8)) &
            (df['funnel_efficiency'] > df['funnel_efficiency'].quantile(0.7))
        ]
        
        patterns['high_performing_campaigns'] = {
            'count': len(high_performers),
            'avg_roas': high_performers['roas'].mean(),
            'avg_cac': high_performers['cac'].mean(),
            'top_channels': high_performers['channel'].value_counts().head(3).to_dict(),
            'top_seo_categories': high_performers['seo_category'].value_counts().head(3).to_dict()
        }
        
        # Underperforming campaigns with potential
        underperformers = df[
            (df['impressions'] > df['impressions'].quantile(0.6)) &
            (df['roas'] < df['roas'].quantile(0.3))
        ]
        
        patterns['underperforming_with_potential'] = {
            'count': len(underperformers),
            'avg_impressions': underperformers['impressions'].mean(),
            'avg_roas': underperformers['roas'].mean(),
            'improvement_areas': {
                'low_ctr': len(underperformers[underperformers['ctr'] < underperformers['ctr'].median()]),
                'low_conversion': len(underperformers[underperformers['signup_rate'] < underperformers['signup_rate'].median()]),
                'poor_retention': len(underperformers[underperformers['retention_rate'] < underperformers['retention_rate'].median()])
            }
        }
        
        # Channel-specific patterns
        patterns['channel_insights'] = {}
        for channel in df['channel'].unique():
            channel_data = df[df['channel'] == channel]
            patterns['channel_insights'][channel] = {
                'best_metric': channel_data[['roas', 'ctr', 'funnel_efficiency']].mean().idxmax(),
                'worst_metric': channel_data[['roas', 'ctr', 'funnel_efficiency']].mean().idxmin(),
                'consistency': channel_data['roas'].std(),  # Lower std = more consistent
                'scalability': channel_data['spend_usd'].max() / channel_data['spend_usd'].mean()
            }
        
        return patterns
    
    def generate_recommendations(self, df: pd.DataFrame, channel_analysis: Dict, 
                               seo_analysis: Dict, patterns: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Budget reallocation recommendations
        best_roas_channel = channel_analysis['best_roas']
        worst_roas_channel = channel_analysis['worst_roas']
        
        recommendations.append({
            'type': 'budget_optimization',
            'priority': 'high',
            'title': 'Reallocate Budget to High-ROAS Channels',
            'description': f"Move budget from {worst_roas_channel} (low ROAS) to {best_roas_channel} (high ROAS)",
            'impact_estimate': 'potential 20-40% ROAS improvement',
            'action_items': [
                f"Reduce {worst_roas_channel} spend by 30%",
                f"Increase {best_roas_channel} spend by equivalent amount",
                "Monitor performance for 2 weeks before further adjustments"
            ]
        })
        
        # SEO optimization recommendations
        if seo_analysis['high_opportunity_categories']:
            top_seo_opportunity = max(seo_analysis['high_opportunity_categories'].items(),
                                    key=lambda x: x[1]['monthly_search_volume'])
            
            recommendations.append({
                'type': 'seo_optimization',
                'priority': 'medium',
                'title': f'Optimize SEO for {top_seo_opportunity[0]} Category',
                'description': f"High search volume ({top_seo_opportunity[1]['monthly_search_volume']:,.0f}) but poor ranking",
                'impact_estimate': 'potential 50-100% organic traffic increase',
                'action_items': [
                    "Conduct keyword gap analysis",
                    "Optimize on-page content and meta descriptions",
                    "Build high-quality backlinks",
                    "Improve page loading speed"
                ]
            })
        
        # Funnel optimization recommendations
        avg_signup_rate = df['signup_rate'].mean()
        avg_retention_rate = df['retention_rate'].mean()
        
        if avg_signup_rate < 0.1:  # Less than 10% signup rate
            recommendations.append({
                'type': 'conversion_optimization',
                'priority': 'high',
                'title': 'Improve Signup Conversion Rate',
                'description': f"Current signup rate ({avg_signup_rate:.1%}) is below industry benchmarks",
                'impact_estimate': 'potential 15-30% signup increase',
                'action_items': [
                    "A/B test signup form design and copy",
                    "Reduce signup friction",
                    "Add social proof and testimonials",
                    "Implement exit-intent popups"
                ]
            })
        
        if avg_retention_rate < 0.3:  # Less than 30% retention
            recommendations.append({
                'type': 'retention_optimization',
                'priority': 'medium',
                'title': 'Enhance Customer Retention',
                'description': f"Retention rate ({avg_retention_rate:.1%}) needs improvement for sustainable growth",
                'impact_estimate': 'potential 25-50% LTV increase',
                'action_items': [
                    "Implement personalized email sequences",
                    "Create loyalty program",
                    "Improve onboarding experience",
                    "Add retargeting campaigns for first-time buyers"
                ]
            })
        
        # Campaign-specific recommendations
        if patterns['underperforming_with_potential']['count'] > 0:
            recommendations.append({
                'type': 'campaign_optimization',
                'priority': 'medium',
                'title': 'Optimize Underperforming High-Volume Campaigns',
                'description': f"{patterns['underperforming_with_potential']['count']} campaigns have high impressions but low ROAS",
                'impact_estimate': 'potential 30-60% ROAS improvement',
                'action_items': [
                    "Audit ad creative and messaging",
                    "Refine audience targeting",
                    "Test different landing pages",
                    "Optimize bid strategies"
                ]
            })
        
        return recommendations
    
    def process_d2c_data(self, filename: str = "D2C_Synthetic_Dataset.csv") -> Dict:
        """Complete D2C data processing pipeline"""
        
        # Load and clean data
        df = self.load_d2c_data(filename)
        df_clean = self.clean_d2c_data(df)
        
        # Calculate metrics
        df_with_metrics = self.calculate_funnel_metrics(df_clean)
        
        # Perform analyses
        channel_analysis = self.analyze_channel_performance(df_with_metrics)
        seo_analysis = self.analyze_seo_opportunities(df_with_metrics)
        patterns = self.identify_growth_patterns(df_with_metrics)
        recommendations = self.generate_recommendations(df_with_metrics, channel_analysis, 
                                                      seo_analysis, patterns)
        
        # Compile results
        results = {
            'data_summary': {
                'total_campaigns': len(df_with_metrics),
                'total_spend': df_with_metrics['spend_usd'].sum(),
                'total_revenue': df_with_metrics['revenue_usd'].sum(),
                'overall_roas': df_with_metrics['revenue_usd'].sum() / df_with_metrics['spend_usd'].sum(),
                'avg_cac': df_with_metrics['cac'].mean(),
                'channels': df_with_metrics['channel'].nunique(),
                'seo_categories': df_with_metrics['seo_category'].nunique()
            },
            'processed_data': df_with_metrics,
            'channel_analysis': channel_analysis,
            'seo_analysis': seo_analysis,
            'growth_patterns': patterns,
            'recommendations': recommendations
        }
        
        return results
