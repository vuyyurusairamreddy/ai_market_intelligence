import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats
import logging

class ConfidenceScorer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_statistical_confidence(self, data: pd.Series, insight_type: str) -> float:
        """Calculate statistical confidence based on data quality and size"""
        
        if data.empty:
            return 0.0
        
        # Base confidence factors
        sample_size_factor = min(len(data) / 1000, 1.0)  # Normalize to 1000 samples
        completeness_factor = 1 - (data.isna().sum() / len(data))  # Data completeness
        
        # Variance factor (lower variance = higher confidence)
        if data.dtype in ['int64', 'float64']:
            cv = stats.variation(data.dropna()) if len(data.dropna()) > 1 else 0
            variance_factor = max(0, 1 - min(cv, 1))  # Cap at 1
        else:
            variance_factor = 0.8  # Default for categorical data
        
        # Insight type weighting
        type_weights = {
            'category_trends': 0.9,
            'growth_opportunities': 0.8,
            'store_comparison': 0.85,
            'market_intelligence': 0.9
        }
        
        type_weight = type_weights.get(insight_type, 0.7)
        
        # Combined confidence score
        confidence = (
            sample_size_factor * 0.3 +
            completeness_factor * 0.3 +
            variance_factor * 0.2 +
            type_weight * 0.2
        )
        
        return round(confidence, 3)
    
    def validate_insight_correlations(self, df: pd.DataFrame, insight_claims: List[str]) -> Dict[str, float]:
        """Validate insight claims against data correlations"""
        validation_scores = {}
        
        # Define validation metrics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_columns].corr()
        
        # Example validation patterns
        validation_patterns = {
            'high_rating_high_reviews': {
                'claim': 'apps with high ratings tend to have more reviews',
                'validation': correlation_matrix.loc['Rating', 'Reviews'] if 'Rating' in correlation_matrix.columns and 'Reviews' in correlation_matrix.columns else 0
            },
            'category_performance_consistency': {
                'claim': 'category performance is consistent across metrics',
                'validation': self._validate_category_consistency(df)
            },
            'store_performance_difference': {
                'claim': 'significant differences exist between stores',
                'validation': self._validate_store_differences(df)
            }
        }
        
        for pattern_name, pattern_data in validation_patterns.items():
            score = abs(pattern_data['validation'])  # Absolute correlation strength
            validation_scores[pattern_name] = min(score, 1.0)
        
        return validation_scores
    
    def _validate_category_consistency(self, df: pd.DataFrame) -> float:
        """Validate consistency of category performance across metrics"""
        if 'Category' not in df.columns:
            return 0.5
        
        category_metrics = df.groupby('Category').agg({
            'Rating': 'mean',
            'Reviews': 'mean',
            'Cross_Platform_Score': 'mean'
        }).fillna(0)
        
        if len(category_metrics) < 2:
            return 0.5
        
        # Calculate correlation between different metrics
        correlations = []
        metric_columns = category_metrics.columns
        
        for i, col1 in enumerate(metric_columns):
            for col2 in metric_columns[i+1:]:
                corr = category_metrics[col1].corr(category_metrics[col2])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.5
    
    def _validate_store_differences(self, df: pd.DataFrame) -> float:
        """Validate that store performance differences are statistically significant"""
        if 'Store' not in df.columns or df['Store'].nunique() < 2:
            return 0.5
        
        stores = df['Store'].unique()
        if len(stores) < 2:
            return 0.5
        
        # T-test for rating differences between stores
        store1_ratings = df[df['Store'] == stores[0]]['Rating'].dropna()
        store2_ratings = df[df['Store'] == stores[1]]['Rating'].dropna()
        
        if len(store1_ratings) < 2 or len(store2_ratings) < 2:
            return 0.5
        
        try:
            t_stat, p_value = stats.ttest_ind(store1_ratings, store2_ratings)
            # Convert p-value to confidence (lower p-value = higher confidence)
            significance_score = max(0, 1 - p_value)
            return significance_score
        except:
            return 0.5
    
    def calculate_insight_confidence(self, insight: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive confidence scores for an insight"""
        
        insight_type = insight.get('type', 'unknown')
        
        # Statistical confidence based on relevant data
        if insight_type == 'category_trends':
            relevant_data = df['Category']
        elif insight_type == 'growth_opportunities':
            relevant_data = df['Cross_Platform_Score']
        elif insight_type == 'store_comparison':
            relevant_data = df['Store']
        else:
            relevant_data = df['Rating']  # Default to rating data
        
        statistical_confidence = self.calculate_statistical_confidence(relevant_data, insight_type)
        
        # Data quality assessment
        data_quality_score = self._assess_data_quality(df)
        
        # Insight content quality (simplified heuristic)
        content_quality_score = self._assess_content_quality(insight.get('content', ''))
        
        # Combined confidence score
        overall_confidence = (
            statistical_confidence * 0.4 +
            data_quality_score * 0.3 +
            content_quality_score * 0.3
        )
        
        # Confidence level categorization
        if overall_confidence >= 0.8:
            confidence_level = 'High'
        elif overall_confidence >= 0.6:
            confidence_level = 'Medium'
        elif overall_confidence >= 0.4:
            confidence_level = 'Low'
        else:
            confidence_level = 'Very Low'
        
        # Add confidence metrics to insight
        confidence_metrics = {
            'overall_confidence': round(overall_confidence, 3),
            'confidence_level': confidence_level,
            'statistical_confidence': round(statistical_confidence, 3),
            'data_quality_score': round(data_quality_score, 3),
            'content_quality_score': round(content_quality_score, 3),
            'sample_size': len(df),
            'data_completeness': round(1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))), 3)
        }
        
        insight['confidence_metrics'] = confidence_metrics
        
        return insight
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Assess overall data quality"""
        
        # Completeness score
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        # Uniqueness score (for key fields)
        key_fields = ['App']
        uniqueness_scores = []
        for field in key_fields:
            if field in df.columns:
                uniqueness = df[field].nunique() / len(df)
                uniqueness_scores.append(uniqueness)
        
        uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 1.0
        
        # Consistency score (simplified)
        consistency = 0.9  # Placeholder - could be enhanced with more complex validation
        
        # Combined data quality score
        quality_score = (completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3)
        
        return quality_score
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of insight content (simplified heuristic)"""
        
        if not content or content == 'Analysis failed':
            return 0.0
        
        # Length factor (reasonable length indicates thoroughness)
        length_score = min(len(content) / 1000, 1.0)  # Normalize to 1000 characters
        
        # Keywords indicating analysis depth
        analysis_keywords = [
            'recommend', 'suggest', 'opportunity', 'trend', 'performance',
            'strategy', 'competitive', 'market', 'growth', 'insight',
            'analysis', 'data', 'evidence', 'correlation', 'significant'
        ]
        
        keyword_count = sum(1 for keyword in analysis_keywords if keyword.lower() in content.lower())
        keyword_score = min(keyword_count / 10, 1.0)  # Normalize to 10 keywords
        
        # Structure indicators (lists, numbers, organization)
        structure_indicators = ['1.', '2.', '3.', '-', '*', 'recommendations', 'insights']
        structure_count = sum(1 for indicator in structure_indicators if indicator in content)
        structure_score = min(structure_count / 5, 1.0)  # Normalize to 5 indicators
        
        # Combined content quality score
        content_quality = (length_score * 0.3 + keyword_score * 0.4 + structure_score * 0.3)
        
        return content_quality
    
    def generate_recommendations(self, insight: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on confidence levels"""
        
        confidence_level = insight.get('confidence_metrics', {}).get('confidence_level', 'Low')
        insight_type = insight.get('type', 'unknown')
        
        recommendations = []
        
        if confidence_level == 'High':
            recommendations.extend([
                "This insight has high statistical confidence and can be used for strategic decision-making",
                "Consider implementing recommendations immediately",
                "Use this insight as a benchmark for similar analyses"
            ])
        elif confidence_level == 'Medium':
            recommendations.extend([
                "This insight shows moderate confidence - validate with additional data if possible",
                "Consider as supporting evidence for decisions",
                "Monitor trends to confirm patterns"
            ])
        elif confidence_level == 'Low':
            recommendations.extend([
                "This insight has limited confidence - use cautiously",
                "Gather more data before making significant decisions",
                "Consider as preliminary finding requiring validation"
            ])
        else:
            recommendations.extend([
                "This insight has very low confidence - not recommended for decision-making",
                "Significant data quality issues detected",
                "Collect more comprehensive data before analysis"
            ])
        
        # Type-specific recommendations
        if insight_type == 'category_trends' and confidence_level in ['High', 'Medium']:
            recommendations.append("Focus development efforts on identified high-opportunity categories")
        elif insight_type == 'growth_opportunities' and confidence_level in ['High', 'Medium']:
            recommendations.append("Prioritize apps and categories identified for growth potential")
        elif insight_type == 'store_comparison' and confidence_level in ['High', 'Medium']:
            recommendations.append("Tailor store-specific strategies based on performance differences")
        
        return recommendations
