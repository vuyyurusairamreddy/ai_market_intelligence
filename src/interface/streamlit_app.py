import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from typing import List, Dict, Any
import numpy as np
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from config.settings import Config
except ImportError:
    st.error("Config module not found. Make sure config/settings.py exists.")
    st.stop()

class MarketIntelligenceApp:
    def __init__(self):
        self.config = Config()
        st.set_page_config(
            page_title="AI-Powered Market Intelligence",
            page_icon="📱",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better formatting
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e6e9ef;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        .insight-container {
            background-color: #f8f9fa;
            border-left: 4px solid #0083B8;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }
        
        .confidence-high {
            color: #28a745;
            font-weight: bold;
        }
        
        .confidence-medium {
            color: #ffc107;
            font-weight: bold;
        }
        
        .confidence-low {
            color: #dc3545;
            font-weight: bold;
        }
        
        .data-source-info {
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.9rem;
            margin: 0.5rem 0;
        }
        
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 1rem 0 0.5rem 0;
        }
        
        .bullet-point {
            margin-left: 1rem;
            margin-bottom: 0.3rem;
        }
        
        .table-container {
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def clean_text_for_display(self, text):
        """Clean text for display purposes"""
        if pd.isna(text):
            return "Unknown"
        
        text = str(text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('Ã©', 'é')
        text = text.replace('Ã¡', 'á')
        text = text.replace('Ã³', 'ó')
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if text else "Unknown"
    
    def format_insight_content(self, content):
        """Format insight content for better Streamlit display"""
        
        # Split content into lines
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                formatted_lines.append('')
                continue
            
            # Format headers (ALL CAPS sections)
            if line.isupper() and len(line) > 10 and ':' not in line:
                formatted_lines.append(f"### {line.title()}")
                
            # Format section headers with colons
            elif line.isupper() and ':' in line:
                formatted_lines.append(f"**{line.title()}**")
                
            # Format bullet points
            elif line.startswith('•'):
                formatted_lines.append(f"- {line[1:].strip()}")
                
            # Format numbered lists
            elif re.match(r'^\d+\.', line):
                formatted_lines.append(f"{line}")
                
            # Format bold text
            elif line.startswith('**') and line.endswith('**'):
                formatted_lines.append(f"{line}")
                
            # Regular paragraphs
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def display_formatted_insight(self, insight, title):
        """Display insight with proper formatting and structure"""
        
        if not insight or not insight.get('content'):
            st.warning("No insights available")
            return
        
        content = insight.get('content', '')
        
        # Create main container
        st.markdown(f"### {title}")
        
        # Split content into sections if it's long
        if len(content) > 2000:
            self.display_sectioned_insight(content, insight)
        else:
            self.display_simple_insight(content, insight)
    
    def display_sectioned_insight(self, content, insight):
        """Display long insights in expandable sections"""
        
        # Define section patterns
        sections = {
            "Overview": ["OVERVIEW", "ECOSYSTEM", "HEALTH"],
            "Analysis": ["ANALYSIS", "LEADERSHIP", "DOMINANCE", "PATTERNS"],
            "Insights": ["INSIGHTS", "FACTORS", "LANDSCAPE"],
            "Opportunities": ["OPPORTUNITIES", "STRATEGIC", "INVESTMENT"],
            "Recommendations": ["RECOMMENDATIONS", "PRIORITIES"],
            "Predictions": ["PREDICTIONS", "ASSESSMENT", "RISK"]
        }
        
        # Split content by sections
        lines = content.split('\n')
        current_section = "Overview"
        section_content = {section: [] for section in sections.keys()}
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Check if line is a section header
            section_found = False
            for section_name, keywords in sections.items():
                if any(keyword in line_upper for keyword in keywords) and len(line_upper) > 10:
                    current_section = section_name
                    section_found = True
                    break
            
            if not section_found:
                section_content[current_section].append(line)
        
        # Display sections in tabs or expanders
        tab_names = [name for name, content_list in section_content.items() if content_list]
        
        if len(tab_names) > 1:
            tabs = st.tabs(tab_names)
            for i, tab_name in enumerate(tab_names):
                with tabs[i]:
                    section_text = '\n'.join(section_content[tab_name])
                    formatted_text = self.format_insight_content(section_text)
                    st.markdown(formatted_text)
        else:
            # Single section - display normally
            formatted_content = self.format_insight_content(content)
            st.markdown(formatted_content)
        
        # Display confidence metrics
        self.display_confidence_metrics(insight)
    
    def display_simple_insight(self, content, insight):
        """Display simple insights with basic formatting"""
        
        # Check for tables in content
        if '|' in content and ('Platform' in content or 'Category' in content):
            self.display_content_with_tables(content)
        else:
            formatted_content = self.format_insight_content(content)
            st.markdown(formatted_content)
        
        # Display confidence metrics
        self.display_confidence_metrics(insight)
    
    def display_content_with_tables(self, content):
        """Display content that contains markdown tables"""
        
        lines = content.split('\n')
        current_block = []
        in_table = False
        
        for line in lines:
            if '|' in line and ('---' in line or any(word in line for word in ['Platform', 'Category', 'Advantages', 'Store'])):
                # Start of table
                if current_block:
                    # Display content before table
                    block_text = '\n'.join(current_block)
                    formatted_text = self.format_insight_content(block_text)
                    st.markdown(formatted_text)
                    current_block = []
                
                in_table = True
                table_lines = [line]
            elif in_table and '|' in line:
                table_lines.append(line)
            elif in_table:
                # End of table
                if table_lines:
                    table_md = '\n'.join(table_lines)
                    st.markdown(table_md)
                in_table = False
                current_block = [line] if line.strip() else []
            else:
                current_block.append(line)
        
        # Display remaining content
        if current_block:
            block_text = '\n'.join(current_block)
            formatted_text = self.format_insight_content(block_text)
            st.markdown(formatted_text)
        
        # Display final table if exists
        if in_table and 'table_lines' in locals():
            table_md = '\n'.join(table_lines)
            st.markdown(table_md)
    
    def display_confidence_metrics(self, insight):
        """Display confidence metrics in a clean format"""
        
        if 'confidence_metrics' not in insight:
            return
        
        confidence = insight['confidence_metrics']
        conf_level = confidence.get('confidence_level', 'Unknown')
        conf_score = confidence.get('overall_confidence', 0)
        sample_size = confidence.get('sample_size', 0)
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_class = "confidence-high" if conf_level == "High" else "confidence-medium" if conf_level == "Medium" else "confidence-low"
            st.markdown(f'<p class="{confidence_class}">Confidence: {conf_level}</p>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Accuracy Score", f"{conf_score:.2f}")
        
        with col3:
            st.metric("Data Points", f"{sample_size:,}")
        
        # Display recommendations if available
        if 'recommendations' in insight and insight['recommendations']:
            with st.expander("💡 Key Recommendations"):
                for i, rec in enumerate(insight['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")
    
    def create_insights_from_data(self):
        """Create insights from actual data using built-in logic"""
        if not hasattr(self, 'df') or self.df.empty:
            return []
        
        try:
            # Load insights engine
            from insights.llm_insights import LLMInsightsEngine
            
            engine = LLMInsightsEngine()
            insights = engine.generate_all_insights(self.df)
            
            return insights
            
        except Exception as e:
            st.sidebar.error(f"Error generating insights: {e}")
            return []
    
    def load_data(self):
        """Load processed data and insights with comprehensive encoding handling"""
        try:
            # Show loading message
            with st.spinner("Loading data..."):
                
                # Load unified dataset with proper encoding
                unified_path = Path(self.config.UNIFIED_DATA_PATH) / self.config.UNIFIED_DATA_FILE
                
                if unified_path.exists():
                    # Try multiple encodings for JSON
                    encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'cp1252', 'latin1']
                    
                    for encoding in encodings:
                        try:
                            with open(unified_path, 'r', encoding=encoding) as f:
                                self.unified_data = json.load(f)
                            
                            st.sidebar.success(f"Data loaded with {encoding} encoding")
                            break
                            
                        except UnicodeDecodeError:
                            continue
                        except json.JSONDecodeError as e:
                            st.sidebar.error(f"JSON decode error with {encoding}: {str(e)[:100]}")
                            continue
                        except Exception as e:
                            st.sidebar.warning(f"Error with {encoding}: {str(e)[:100]}")
                            continue
                    else:
                        st.error("Could not read unified dataset with any encoding")
                        st.info("Please run: python main.py")
                        return False
                    
                    # Convert to DataFrame
                    try:
                        self.df = pd.DataFrame(self.unified_data['data'])
                        
                        if self.df.empty:
                            st.error("Loaded dataset is empty")
                            return False
                        
                        # Clean text columns in loaded data
                        text_columns = ['App', 'Category', 'Genres', 'Content Rating']
                        for col in text_columns:
                            if col in self.df.columns:
                                self.df[col] = self.df[col].apply(self.clean_text_for_display)
                        
                        # Ensure numeric columns are properly typed
                        numeric_columns = ['Rating', 'Reviews', 'Price']
                        for col in numeric_columns:
                            if col in self.df.columns:
                                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                        
                        # Ensure required columns exist
                        if 'Cross_Platform_Score' not in self.df.columns:
                            st.sidebar.warning("Adding Cross_Platform_Score from actual data...")
                            self.add_cross_platform_score()
                        
                        if 'Store' not in self.df.columns:
                            self.df['Store'] = 'Google Play'
                            st.sidebar.info("Added Store column")
                        
                    except Exception as e:
                        st.error(f"Error processing DataFrame: {e}")
                        return False
                        
                else:
                    st.error("Unified dataset not found")
                    st.markdown("""
                    **To fix this:**
                    1. Run: `python main.py` 
                    2. Or run: `python fix_actual_data.py`
                    """)
                    return False
                
                # Load insights - IMPROVED LOGIC
                self.insights = []
                insights_loaded = False
                
                insights_path = Path(self.config.OUTPUTS_PATH) / self.config.INSIGHTS_FILE
                
                # Try to load existing insights file
                if insights_path.exists():
                    for encoding in encodings:
                        try:
                            with open(insights_path, 'r', encoding=encoding) as f:
                                insights_data = json.load(f)
                            
                            if isinstance(insights_data, dict) and 'insights' in insights_data:
                                self.insights = insights_data['insights']
                                insights_loaded = True
                                st.sidebar.success(f"Insights loaded successfully ({len(self.insights)} insights)")
                                break
                            elif isinstance(insights_data, list):
                                self.insights = insights_data
                                insights_loaded = True
                                st.sidebar.success(f"Insights loaded successfully ({len(self.insights)} insights)")
                                break
                                
                        except UnicodeDecodeError:
                            continue
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            continue
                
                # If insights not loaded, create them from data
                if not insights_loaded:
                    st.sidebar.warning("Insights not found - generating from data...")
                    try:
                        self.insights = self.create_insights_from_data()
                        if self.insights:
                            st.sidebar.success(f"Generated {len(self.insights)} insights from data")
                        else:
                            st.sidebar.error("Failed to generate insights")
                            self.insights = []
                    except Exception as e:
                        st.sidebar.error(f"Error generating insights: {e}")
                        self.insights = []
            
            return True
            
        except Exception as e:
            st.error(f"Critical error loading data: {e}")
            st.markdown("""
            **Troubleshooting steps:**
            1. Check if your data files exist in `data/raw/`
            2. Run: `python fix_encoding_issues.py`
            3. Run: `python main.py`
            4. Restart Streamlit
            """)
            return False
    
    def add_cross_platform_score(self):
        """Add Cross_Platform_Score using actual data only"""
        try:
            # Ensure numeric columns
            self.df['Rating'] = pd.to_numeric(self.df['Rating'], errors='coerce').fillna(0)
            self.df['Reviews'] = pd.to_numeric(self.df['Reviews'], errors='coerce').fillna(0)
            
            # Calculate based on actual data
            rating_normalized = self.df['Rating'] / 5.0
            reviews_log = np.log1p(self.df['Reviews'])
            
            if reviews_log.max() > 0:
                reviews_normalized = reviews_log / reviews_log.max()
            else:
                reviews_normalized = pd.Series([0] * len(self.df))
            
            self.df['Cross_Platform_Score'] = ((rating_normalized * 0.6 + reviews_normalized * 0.4) * 100).round(2)
            
            # Add Market Position
            def get_position(score):
                if score >= self.df['Cross_Platform_Score'].quantile(0.9):
                    return 'Leader'
                elif score >= self.df['Cross_Platform_Score'].quantile(0.7):
                    return 'Strong'
                elif score >= self.df['Cross_Platform_Score'].quantile(0.3):
                    return 'Average'
                else:
                    return 'Weak'
            
            self.df['Market_Position'] = self.df['Cross_Platform_Score'].apply(get_position)
            
            st.sidebar.success("Cross_Platform_Score calculated from actual data")
            
        except Exception as e:
            st.sidebar.error(f"Error calculating Cross_Platform_Score: {e}")
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # Data source information
        if hasattr(self, 'df') and not self.df.empty:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Data Summary")
            st.sidebar.info(f"**Apps:** {len(self.df):,}")
            
            if 'Category' in self.df.columns:
                st.sidebar.info(f"**Categories:** {self.df['Category'].nunique()}")
            
            if 'Store' in self.df.columns:
                stores = self.df['Store'].unique()
                st.sidebar.info(f"**Stores:** {', '.join(stores)}")
            
            # Data quality indicator
            if hasattr(self, 'insights') and self.insights:
                st.sidebar.success("**AI Insights:** Available")
            else:
                st.sidebar.warning("**AI Insights:** Limited")
        
        st.sidebar.markdown("---")
        
        pages = [
            "Dashboard Overview",
            "Category Analysis", 
            "Store Comparison",
            "Growth Opportunities",
            "Market Intelligence",
            "Query Interface"
        ]
        
        return st.sidebar.selectbox("Select Page", pages)
    
    def render_dashboard_overview(self):
        """Render main dashboard overview using actual data"""
        st.title("AI-Powered Market Intelligence Dashboard")
        
        # Data source disclaimer
        st.markdown("""
        <div class="data-source-info">
        <strong>Data Source:</strong> Analysis based on your actual Google Play Store dataset - no synthetic data used
        </div>
        """, unsafe_allow_html=True)
        
        if not hasattr(self, 'df') or self.df.empty:
            st.error("No data available")
            return
        
        # Key metrics from actual data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_apps = len(self.df)
            st.metric("Total Apps", f"{total_apps:,}", help="From your actual dataset")
        
        with col2:
            avg_rating = self.df['Rating'].mean() if 'Rating' in self.df.columns else 0
            delta_rating = "High Quality" if avg_rating >= 4.0 else "Average" if avg_rating >= 3.5 else "Below Average"
            st.metric("Avg Rating", f"{avg_rating:.2f}", delta=delta_rating)
        
        with col3:
            total_reviews = self.df['Reviews'].sum() if 'Reviews' in self.df.columns else 0
            st.metric("Total Reviews", f"{total_reviews:,.0f}", help="Sum of actual review counts")
        
        with col4:
            categories = self.df['Category'].nunique() if 'Category' in self.df.columns else 0
            st.metric("Categories", f"{categories}", help="Unique categories in your data")
        
        st.markdown("---")
        
        # Charts based on actual data
        st.subheader("Market Overview (Your Actual Data)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Category' in self.df.columns:
                # Category distribution from actual data
                category_counts = self.df['Category'].value_counts().head(10)
                fig_cat = px.bar(
                    x=category_counts.values,
                    y=[self.clean_text_for_display(cat) for cat in category_counts.index],
                    orientation='h',
                    title="Top 10 Categories by App Count",
                    labels={'x': 'Number of Apps', 'y': 'Category'},
                    color=category_counts.values,
                    color_continuous_scale='Blues'
                )
                fig_cat.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Category data not available")
        
        with col2:
            # Store distribution from actual data
            if 'Store' in self.df.columns:
                store_counts = self.df['Store'].value_counts()
                fig_store = px.pie(
                    values=store_counts.values,
                    names=store_counts.index,
                    title="Apps by Store (Your Data)",
                    color_discrete_sequence=['#0083B8', '#00A2D4', '#4FC3F7']
                )
                fig_store.update_layout(height=400)
                st.plotly_chart(fig_store, use_container_width=True)
            else:
                # Single platform visualization
                st.info("Single platform analysis")
                fig_single = go.Figure(data=[
                    go.Bar(x=['Google Play'], y=[len(self.df)], marker_color='#0083B8')
                ])
                fig_single.update_layout(
                    title="Platform Coverage",
                    xaxis_title="Platform",
                    yaxis_title="Number of Apps",
                    height=400
                )
                st.plotly_chart(fig_single, use_container_width=True)
        
        # Rating and Review Analysis
        st.subheader("Quality & Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Rating' in self.df.columns:
                # Rating distribution from actual data
                fig_ratings = px.histogram(
                    self.df,
                    x='Rating',
                    nbins=20,
                    title="App Rating Distribution",
                    labels={'Rating': 'Rating (1-5 scale)', 'count': 'Number of Apps'},
                    color_discrete_sequence=['#0083B8']
                )
                fig_ratings.update_layout(height=350)
                st.plotly_chart(fig_ratings, use_container_width=True)
            else:
                st.info("Rating data not available")
        
        with col2:
            if 'Reviews' in self.df.columns:
                # Review distribution (log scale for better visualization)
                df_with_reviews = self.df[self.df['Reviews'] > 0]
                if not df_with_reviews.empty:
                    fig_reviews = px.histogram(
                        df_with_reviews,
                        x='Reviews',
                        nbins=20,
                        title="Review Count Distribution (Apps with Reviews)",
                        labels={'Reviews': 'Number of Reviews', 'count': 'Number of Apps'},
                        color_discrete_sequence=['#00A2D4']
                    )
                    fig_reviews.update_layout(
                        xaxis_type="log",
                        height=350
                    )
                    st.plotly_chart(fig_reviews, use_container_width=True)
                else:
                    st.info("No review data available")
            else:
                st.info("Review data not available")
        
        # Performance Insights
        if 'Cross_Platform_Score' in self.df.columns:
            st.subheader("Performance Insights")
            
            # Performance score distribution
            fig_performance = px.histogram(
                self.df,
                x='Cross_Platform_Score',
                nbins=25,
                title="Cross-Platform Performance Score Distribution",
                labels={'Cross_Platform_Score': 'Performance Score (0-100)', 'count': 'Number of Apps'},
                color_discrete_sequence=['#4FC3F7']
            )
            fig_performance.update_layout(height=350)
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Top performers
            top_performers = self.df.nlargest(10, 'Cross_Platform_Score')[
                ['App', 'Category', 'Rating', 'Reviews', 'Cross_Platform_Score']
            ]
            
            st.subheader("Top Performing Apps")
            
            # Clean text in top performers
            display_df = top_performers.copy()
            display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
            display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
            
            st.dataframe(
                display_df.style.format({
                    'Rating': '{:.1f}',
                    'Reviews': '{:,.0f}',
                    'Cross_Platform_Score': '{:.1f}'
                }),
                use_container_width=True
            )
    
    def render_category_analysis(self):
        """Render category analysis using actual data"""
        st.title("Category Analysis")
        st.caption("Based on your actual app dataset")
        
        if not hasattr(self, 'df') or self.df.empty:
            st.error("No data available")
            return
        
        if 'Category' in self.df.columns:
            # Category selector from actual data
            categories = sorted([self.clean_text_for_display(cat) for cat in self.df['Category'].unique()])
            selected_category = st.selectbox("Select Category", categories)
            
            # Filter actual data
            category_data = self.df[self.df['Category'].apply(self.clean_text_for_display) == selected_category]
            
            if category_data.empty:
                st.warning(f"No data available for {selected_category}")
                return
            
            # Category metrics from actual data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Apps in Category", len(category_data))
            
            with col2:
                if 'Rating' in category_data.columns:
                    avg_rating = category_data['Rating'].mean()
                    st.metric("Avg Rating", f"{avg_rating:.2f}")
                else:
                    st.metric("Avg Rating", "N/A")
            
            with col3:
                if 'Reviews' in category_data.columns:
                    avg_reviews = category_data['Reviews'].mean()
                    st.metric("Avg Reviews", f"{avg_reviews:,.0f}")
                else:
                    st.metric("Avg Reviews", "N/A")
            
            with col4:
                if 'Cross_Platform_Score' in category_data.columns:
                    avg_score = category_data['Cross_Platform_Score'].mean()
                    st.metric("Avg Performance Score", f"{avg_score:.1f}")
                else:
                    st.metric("Avg Performance Score", "N/A")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Rating' in category_data.columns and 'Reviews' in category_data.columns:
                    # Rating vs Reviews scatter
                    fig_scatter = px.scatter(
                        category_data.head(100),  # Limit for performance
                        x='Rating',
                        y='Reviews',
                        title=f"Rating vs Reviews in {selected_category}",
                        labels={'Rating': 'App Rating', 'Reviews': 'Review Count'},
                        color_discrete_sequence=['#0083B8']
                    )
                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                if 'Cross_Platform_Score' in category_data.columns:
                    # Performance score distribution
                    fig_perf = px.histogram(
                        category_data,
                        x='Cross_Platform_Score',
                        nbins=15,
                        title=f"Performance Distribution - {selected_category}",
                        color_discrete_sequence=['#00A2D4']
                    )
                    fig_perf.update_layout(height=400)
                    st.plotly_chart(fig_perf, use_container_width=True)
            
            # Top apps in category from actual data
            st.subheader(f"Top Apps in {selected_category}")
            
            sort_column = 'Cross_Platform_Score' if 'Cross_Platform_Score' in category_data.columns else 'Rating'
            top_apps = category_data.nlargest(15, sort_column)
            
            display_columns = ['App', 'Rating', 'Reviews']
            if 'Cross_Platform_Score' in top_apps.columns:
                display_columns.append('Cross_Platform_Score')
            if 'Store' in top_apps.columns:
                display_columns.append('Store')
            
            # Clean text for display
            display_df = top_apps[display_columns].copy()
            display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
            
            st.dataframe(
                display_df.style.format({
                    'Rating': '{:.1f}',
                    'Reviews': '{:,.0f}',
                    'Cross_Platform_Score': '{:.1f}' if 'Cross_Platform_Score' in display_columns else None
                }),
                use_container_width=True
            )
            
        else:
            st.error("Category data not available in dataset")
        
        # Category insights from actual analysis - IMPROVED FORMATTING
        if hasattr(self, 'insights') and self.insights:
            category_insight = next(
                (insight for insight in self.insights if insight.get('type') == 'category_trends'),
                None
            )
            
            if category_insight and category_insight.get('content'):
                self.display_formatted_insight(category_insight, "AI Category Insights")
    
    def render_store_comparison(self):
        """Render store comparison using only actual data"""
        st.title("Store Comparison")
        st.caption("Analysis of your actual dataset - no synthetic data")
        
        if not hasattr(self, 'df') or self.df.empty:
            st.error("No data available")
            return
        
        if 'Store' not in self.df.columns or self.df['Store'].nunique() == 1:
            st.warning("Single store data detected")
            st.info("Your dataset appears to be single-platform. Showing comprehensive category analysis instead.")
            
            # Show category performance analysis
            st.subheader("Category Performance Analysis")
            
            if 'Category' in self.df.columns and 'Rating' in self.df.columns:
                category_performance = self.df.groupby('Category').agg({
                    'Rating': 'mean',
                    'Reviews': 'mean' if 'Reviews' in self.df.columns else lambda x: 0,
                    'App': 'count'
                }).round(2).sort_values('Rating', ascending=False)
                
                category_performance.columns = ['Avg Rating', 'Avg Reviews', 'App Count']
                
                # Clean category names
                category_performance.index = [self.clean_text_for_display(cat) for cat in category_performance.index]
                
                st.dataframe(
                    category_performance.style.format({
                        'Avg Rating': '{:.2f}',
                        'Avg Reviews': '{:,.0f}',
                        'App Count': '{:,}'
                    }),
                    use_container_width=True
                )
                
                # Visualize category performance
                fig_cat_perf = px.scatter(
                    x=category_performance['Avg Rating'],
                    y=category_performance['Avg Reviews'],
                    size=category_performance['App Count'],
                    hover_name=category_performance.index,
                    title="Category Performance: Rating vs Reviews",
                    labels={'x': 'Average Rating', 'y': 'Average Reviews'},
                    color_discrete_sequence=['#0083B8']
                )
                fig_cat_perf.update_layout(height=500)
                st.plotly_chart(fig_cat_perf, use_container_width=True)
            
        else:
            # Multi-store analysis
            stores = self.df['Store'].unique()
            st.info(f"Found {len(stores)} store(s): {', '.join(stores)}")
            
            # Store metrics from actual data
            store_metrics = self.df.groupby('Store').agg({
                'Rating': 'mean',
                'Reviews': 'mean', 
                'App': 'count'
            }).round(2)
            store_metrics.columns = ['Avg Rating', 'Avg Reviews', 'App Count']
            
            st.subheader("Store Performance Comparison")
            st.dataframe(
                store_metrics.style.format({
                    'Avg Rating': '{:.2f}',
                    'Avg Reviews': '{:,.0f}',
                    'App Count': '{:,}'
                }),
                use_container_width=True
            )
            
            # Visual comparison using actual data
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rating = px.box(
                    self.df,
                    x='Store',
                    y='Rating',
                    title="Rating Distribution by Store",
                    color='Store',
                    color_discrete_sequence=['#0083B8', '#00A2D4']
                )
                fig_rating.update_layout(height=400)
                st.plotly_chart(fig_rating, use_container_width=True)
            
            with col2:
                fig_reviews = px.box(
                    self.df[self.df['Reviews'] > 0],  # Only apps with reviews
                    x='Store', 
                    y='Reviews',
                    title="Review Count Distribution by Store",
                    color='Store',
                    color_discrete_sequence=['#0083B8', '#00A2D4']
                )
                fig_reviews.update_layout(
                    yaxis_type="log",
                    height=400
                )
                st.plotly_chart(fig_reviews, use_container_width=True)
        
        # Store insights - IMPROVED FORMATTING
        if hasattr(self, 'insights') and self.insights:
            store_insight = next(
                (insight for insight in self.insights if insight.get('type') == 'store_comparison'),
                None
            )
            
            if store_insight and store_insight.get('content'):
                self.display_formatted_insight(store_insight, "Store Performance Analysis")
    
    def render_growth_opportunities(self):
        """Render growth opportunities from actual data"""
        st.title("Growth Opportunities")
        st.caption("Identified from your actual dataset")
        
        if not hasattr(self, 'df') or self.df.empty:
            st.error("No data available")
            return
        
        # Find growth opportunities in actual data
        st.subheader("High Potential Apps in Your Data")
        
        if 'Reviews' in self.df.columns and 'Rating' in self.df.columns:
            # Apps with high reviews but lower ratings
            high_potential = self.df[
                (self.df['Reviews'] > self.df['Reviews'].quantile(0.7)) & 
                (self.df['Rating'] < self.df['Rating'].quantile(0.4))
            ]
            
            display_columns = ['App', 'Category', 'Rating', 'Reviews']
            if 'Store' in self.df.columns:
                display_columns.append('Store')
            if 'Cross_Platform_Score' in self.df.columns:
                display_columns.append('Cross_Platform_Score')
            
            if not high_potential.empty:
                # Clean text for display
                display_df = high_potential[display_columns].head(15).copy()
                display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
                display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
                
                st.dataframe(
                    display_df.style.format({
                        'Rating': '{:.1f}',
                        'Reviews': '{:,.0f}',
                        'Cross_Platform_Score': '{:.1f}' if 'Cross_Platform_Score' in display_columns else None
                    }),
                    use_container_width=True
                )
                
                st.success(f"Found {len(high_potential)} apps with high engagement but improvement potential")
            else:
                st.info("No clear underperforming high-engagement apps found in your dataset")
        
        # Category opportunities from actual data
        st.subheader("Category Opportunity Analysis")
        
        if 'Category' in self.df.columns and 'Rating' in self.df.columns and 'Reviews' in self.df.columns:
            category_performance = self.df.groupby('Category').agg({
                'Rating': 'mean',
                'Reviews': 'mean',
                'App': 'count'
            }).round(2)
            category_performance.columns = ['Avg Rating', 'Avg Reviews', 'App Count']
            
            # Calculate opportunity score
            max_reviews = category_performance['Avg Reviews'].max()
            if max_reviews > 0:
                category_performance['Opportunity_Score'] = (
                    category_performance['Avg Reviews'] / max_reviews * 0.6 +
                    (5 - category_performance['Avg Rating']) / 5 * 0.4
                ) * 100
            else:
                category_performance['Opportunity_Score'] = 0
            
            # Clean category names
            category_performance.index = [self.clean_text_for_display(cat) for cat in category_performance.index]
            
            opportunities = category_performance.sort_values('Opportunity_Score', ascending=False).head(10)
            
            st.dataframe(
                opportunities.style.format({
                    'Avg Rating': '{:.2f}',
                    'Avg Reviews': '{:,.0f}',
                    'App Count': '{:,}',
                    'Opportunity_Score': '{:.1f}'
                }),
                use_container_width=True
            )
            
            # Opportunity visualization
            fig_opp = px.scatter(
                x=category_performance['Avg Rating'],
                y=category_performance['Avg Reviews'], 
                size=category_performance['App Count'],
                hover_name=category_performance.index,
                title="Category Opportunity Map",
                labels={
                    'x': 'Average Rating',
                    'y': 'Average Reviews'
                },
                color_discrete_sequence=['#0083B8']
            )
            
            # Add reference lines
            avg_rating = category_performance['Avg Rating'].mean()
            avg_reviews = category_performance['Avg Reviews'].mean()
            
            fig_opp.add_hline(y=avg_reviews, line_dash="dash", line_color="red", 
                             annotation_text="Avg Reviews")
            fig_opp.add_vline(x=avg_rating, line_dash="dash", line_color="red", 
                             annotation_text="Avg Rating")
            
            fig_opp.update_layout(height=500)
            st.plotly_chart(fig_opp, use_container_width=True)
            
            st.info("Categories in the top-left quadrant (high reviews, lower ratings) represent improvement opportunities!")
        
        # Growth insights - IMPROVED FORMATTING
        if hasattr(self, 'insights') and self.insights:
            growth_insight = next(
                (insight for insight in self.insights if insight.get('type') == 'growth_opportunities'),
                None
            )
            
            if growth_insight and growth_insight.get('content'):
                self.display_formatted_insight(growth_insight, "AI Growth Insights")
    
    def render_market_intelligence(self):
        """Render market intelligence from actual data"""
        st.title("Market Intelligence")
        st.caption("Strategic insights from your actual dataset")
        
        if not hasattr(self, 'df') or self.df.empty:
            st.error("No data available")
            return
        
        # Market overview from actual data
        st.subheader("Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_apps = len(self.df)
            st.metric("Total Apps", f"{total_apps:,}")
        
        with col2:
            if 'Store' in self.df.columns:
                android_apps = len(self.df[self.df['Store'] == 'Google Play'])
                st.metric("Android Apps", f"{android_apps:,}")
            else:
                st.metric("Platform", "Google Play")
        
        with col3:
            if 'Store' in self.df.columns:
                ios_apps = len(self.df[self.df['Store'] == 'App Store'])
                st.metric("iOS Apps", f"{ios_apps:,}")
            else:
                categories = self.df['Category'].nunique() if 'Category' in self.df.columns else 0
                st.metric("Categories", f"{categories}")
        
        with col4:
            if 'Rating' in self.df.columns:
                avg_rating = self.df['Rating'].mean()
                st.metric("Market Health", f"{avg_rating:.2f}/5.0")
            else:
                st.metric("Data Quality", "High")
        
        # Market leaders from actual data
        st.subheader("Market Leaders (Top Apps)")
        
        if 'Cross_Platform_Score' in self.df.columns:
            leaders = self.df.nlargest(15, 'Cross_Platform_Score')
            display_columns = ['App', 'Category', 'Rating', 'Reviews', 'Cross_Platform_Score']
        else:
            leaders = self.df.nlargest(15, 'Rating')
            display_columns = ['App', 'Category', 'Rating', 'Reviews']
        
        if 'Store' in self.df.columns:
            display_columns.append('Store')
        
        # Clean text for display
        display_df = leaders[display_columns].copy()
        display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
        display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
        
        st.dataframe(
            display_df.style.format({
                'Rating': '{:.1f}',
                'Reviews': '{:,.0f}',
                'Cross_Platform_Score': '{:.1f}' if 'Cross_Platform_Score' in display_columns else None
            }),
            use_container_width=True
        )
        
        # Market trends visualization
        st.subheader("Market Trends Analysis")
        
        if 'Rating' in self.df.columns and 'Reviews' in self.df.columns and 'Category' in self.df.columns:
            # Sample for performance if dataset is large
            sample_size = min(1000, len(self.df))
            sample_df = self.df.sample(sample_size) if len(self.df) > 1000 else self.df
            
            # Clean category names for visualization
            sample_df = sample_df.copy()
            sample_df['Category_Clean'] = sample_df['Category'].apply(self.clean_text_for_display)
            
            fig_trends = px.scatter(
                sample_df,
                x='Rating',
                y='Reviews',
                color='Category_Clean',
                size='Reviews',
                hover_data=['App'],
                title=f"Rating vs Reviews by Category (Sample of {len(sample_df)} apps)",
                labels={
                    'Rating': 'App Rating',
                    'Reviews': 'Review Count',
                    'Category_Clean': 'Category'
                }
            )
            fig_trends.update_layout(
                yaxis_type="log",
                height=600
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        
        # Market intelligence insights - IMPROVED FORMATTING
        if hasattr(self, 'insights') and self.insights:
            market_insight = next(
                (insight for insight in self.insights if insight.get('type') == 'market_intelligence'),
                None
            )
            
            if market_insight and market_insight.get('content'):
                self.display_formatted_insight(market_insight, "Strategic Market Intelligence")
        else:
            st.info("AI insights not available. Statistical analysis provided above.")
    
    def render_query_interface(self):
        """Render interactive query interface using actual data"""
        st.title("Query Interface")
        st.caption("Interactive analysis of your actual dataset")
        
        if not hasattr(self, 'df') or self.df.empty:
            st.error("No data available")
            return
        
        st.subheader("Quick Market Insights")
        
        # Predefined analysis options
        analysis_options = [
            "Top categories by app count",
            "Highest rated apps in dataset", 
            "Most reviewed apps",
            "Category performance comparison",
            "Apps with improvement potential",
            "Market leaders analysis"
        ]
        
        selected_analysis = st.selectbox("Select Analysis Type", analysis_options)
        
        if st.button("Run Analysis", type="primary"):
            self.execute_predefined_analysis(selected_analysis)
        
        st.markdown("---")
        
        # Custom query builder
        st.subheader("Custom Query Builder")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Available metrics based on actual data
            available_metrics = []
            if 'Rating' in self.df.columns:
                available_metrics.append('Rating')
            if 'Reviews' in self.df.columns:
                available_metrics.append('Reviews')
            if 'Cross_Platform_Score' in self.df.columns:
                available_metrics.append('Cross_Platform_Score')
            
            if available_metrics:
                metric = st.selectbox("Select Metric", available_metrics)
                operation = st.selectbox("Operation", ['Top', 'Bottom', 'Average'])
            else:
                st.error("No numeric metrics available")
                return
        
        with col2:
            filters = {}
            
            if 'Category' in self.df.columns:
                categories = [self.clean_text_for_display(cat) for cat in sorted(self.df['Category'].unique())]
                selected_categories = st.multiselect("Filter by Categories", categories)
                if selected_categories:
                    filters['Category'] = selected_categories
            
            if 'Store' in self.df.columns and self.df['Store'].nunique() > 1:
                stores = sorted(self.df['Store'].unique())
                selected_stores = st.multiselect("Filter by Stores", stores)
                if selected_stores:
                    filters['Store'] = selected_stores
        
        limit = st.slider("Number of Results", 5, 50, 10)
        
        if st.button("Execute Custom Query", type="secondary"):
            if available_metrics:
                self.execute_custom_query(metric, operation, filters, limit)
    
    def execute_predefined_analysis(self, analysis: str):
        """Execute predefined analysis on actual data"""
        
        try:
            if "top categories by app count" in analysis.lower():
                st.subheader("Top Categories by App Count")
                
                if 'Category' in self.df.columns:
                    category_counts = self.df['Category'].value_counts().head(10)
                    
                    # Clean category names
                    clean_counts = pd.Series(
                        category_counts.values,
                        index=[self.clean_text_for_display(cat) for cat in category_counts.index]
                    )
                    
                    st.dataframe(
                        clean_counts.to_frame('App Count').style.format({'App Count': '{:,}'}),
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig = px.bar(
                        x=clean_counts.values,
                        y=clean_counts.index,
                        orientation='h',
                        title="Top Categories by App Count",
                        color_discrete_sequence=['#0083B8']
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Category data not available")
                
            elif "highest rated" in analysis.lower():
                st.subheader("Highest Rated Apps")
                
                if 'Rating' in self.df.columns:
                    top_rated = self.df.nlargest(15, 'Rating')
                    
                    display_columns = ['App', 'Rating']
                    if 'Category' in self.df.columns:
                        display_columns.insert(1, 'Category')
                    if 'Reviews' in self.df.columns:
                        display_columns.append('Reviews')
                    if 'Store' in self.df.columns:
                        display_columns.append('Store')
                    
                    # Clean text
                    display_df = top_rated[display_columns].copy()
                    display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
                    if 'Category' in display_df.columns:
                        display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
                    
                    st.dataframe(
                        display_df.style.format({
                            'Rating': '{:.1f}',
                            'Reviews': '{:,.0f}' if 'Reviews' in display_columns else None
                        }),
                        use_container_width=True
                    )
                else:
                    st.error("Rating data not available")
                
            elif "most reviewed" in analysis.lower():
                st.subheader("Most Reviewed Apps")
                
                if 'Reviews' in self.df.columns:
                    most_reviewed = self.df.nlargest(15, 'Reviews')
                    
                    display_columns = ['App', 'Reviews']
                    if 'Category' in self.df.columns:
                        display_columns.insert(1, 'Category')
                    if 'Rating' in self.df.columns:
                        display_columns.insert(-1, 'Rating')
                    if 'Store' in self.df.columns:
                        display_columns.append('Store')
                    
                    # Clean text
                    display_df = most_reviewed[display_columns].copy()
                    display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
                    if 'Category' in display_df.columns:
                        display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
                    
                    st.dataframe(
                        display_df.style.format({
                            'Reviews': '{:,.0f}',
                            'Rating': '{:.1f}' if 'Rating' in display_columns else None
                        }),
                        use_container_width=True
                    )
                else:
                    st.error("Reviews data not available")
                
            elif "performance comparison" in analysis.lower():
                st.subheader("Category Performance Comparison")
                
                if 'Category' in self.df.columns and 'Rating' in self.df.columns:
                    comparison = self.df.groupby('Category').agg({
                        'Rating': 'mean',
                        'Reviews': 'mean' if 'Reviews' in self.df.columns else lambda x: 0,
                        'App': 'count'
                    }).round(2).sort_values('Rating', ascending=False)
                    
                    comparison.columns = ['Avg Rating', 'Avg Reviews', 'App Count']
                    comparison.index = [self.clean_text_for_display(cat) for cat in comparison.index]
                    
                    st.dataframe(
                        comparison.style.format({
                            'Avg Rating': '{:.2f}',
                            'Avg Reviews': '{:,.0f}',
                            'App Count': '{:,}'
                        }),
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig = px.scatter(
                        x=comparison['Avg Rating'],
                        y=comparison['Avg Reviews'],
                        size=comparison['App Count'],
                        hover_name=comparison.index,
                        title="Category Performance: Rating vs Reviews",
                        color_discrete_sequence=['#0083B8']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Required data not available")
                
            elif "improvement potential" in analysis.lower():
                st.subheader("Apps with Improvement Potential")
                
                if 'Reviews' in self.df.columns and 'Rating' in self.df.columns:
                    # High engagement, room for rating improvement
                    improvement_potential = self.df[
                        (self.df['Reviews'] > self.df['Reviews'].quantile(0.6)) &
                        (self.df['Rating'] < self.df['Rating'].median())
                    ].sort_values('Reviews', ascending=False).head(15)
                    
                    if not improvement_potential.empty:
                        display_columns = ['App', 'Category', 'Rating', 'Reviews']
                        if 'Store' in improvement_potential.columns:
                            display_columns.append('Store')
                        
                        # Clean text
                        display_df = improvement_potential[display_columns].copy()
                        display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
                        if 'Category' in display_df.columns:
                            display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
                        
                        st.dataframe(
                            display_df.style.format({
                                'Rating': '{:.1f}',
                                'Reviews': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                        
                        st.success(f"Found {len(improvement_potential)} apps with high engagement but rating below median ({self.df['Rating'].median():.1f})")
                    else:
                        st.info("No apps found matching improvement criteria")
                else:
                    st.error("Reviews and Rating data required")
                    
            elif "market leaders" in analysis.lower():
                st.subheader("Market Leaders Analysis")
                
                if 'Cross_Platform_Score' in self.df.columns:
                    leaders = self.df.nlargest(20, 'Cross_Platform_Score')
                    
                    display_columns = ['App', 'Category', 'Rating', 'Reviews', 'Cross_Platform_Score']
                    if 'Store' in self.df.columns:
                        display_columns.append('Store')
                    
                    # Clean text
                    display_df = leaders[display_columns].copy()
                    display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
                    display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
                    
                    st.dataframe(
                        display_df.style.format({
                            'Rating': '{:.1f}',
                            'Reviews': '{:,.0f}',
                            'Cross_Platform_Score': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Leader insights
                    avg_leader_rating = leaders['Rating'].mean()
                    avg_leader_reviews = leaders['Reviews'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Leader Rating", f"{avg_leader_rating:.2f}")
                    with col2:
                        st.metric("Avg Leader Reviews", f"{avg_leader_reviews:,.0f}")
                    with col3:
                        st.metric("Performance Threshold", f"{leaders['Cross_Platform_Score'].min():.1f}")
                        
                elif 'Rating' in self.df.columns:
                    st.info("Using Rating for market leader analysis (Cross_Platform_Score not available)")
                    leaders = self.df.nlargest(20, 'Rating')
                    
                    display_columns = ['App', 'Rating']
                    if 'Category' in self.df.columns:
                        display_columns.insert(1, 'Category')
                    if 'Reviews' in self.df.columns:
                        display_columns.append('Reviews')
                    
                    # Clean text
                    display_df = leaders[display_columns].copy()
                    display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
                    if 'Category' in display_df.columns:
                        display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
                    
                    st.dataframe(
                        display_df.style.format({
                            'Rating': '{:.1f}',
                            'Reviews': '{:,.0f}' if 'Reviews' in display_columns else None
                        }),
                        use_container_width=True
                    )
                else:
                    st.error("No suitable metrics available for market leader analysis")
                    
        except Exception as e:
            st.error(f"Error executing analysis: {e}")
    
    def execute_custom_query(self, metric: str, operation: str, filters: Dict, limit: int):
        """Execute custom query on actual data"""
        
        try:
            # Filter data
            filtered_df = self.df.copy()
            
            for filter_col, filter_values in filters.items():
                if filter_col == 'Category':
                    # Handle category filtering with cleaned names
                    original_categories = []
                    for clean_cat in filter_values:
                        # Find original category name
                        for orig_cat in self.df['Category'].unique():
                            if self.clean_text_for_display(orig_cat) == clean_cat:
                                original_categories.append(orig_cat)
                                break
                    filtered_df = filtered_df[filtered_df[filter_col].isin(original_categories)]
                else:
                    filtered_df = filtered_df[filtered_df[filter_col].isin(filter_values)]
            
            if filtered_df.empty:
                st.warning("No data matches the selected filters")
                return
            
            # Apply operation
            if operation == 'Top':
                results = filtered_df.nlargest(limit, metric)
            elif operation == 'Bottom':
                results = filtered_df.nsmallest(limit, metric)
            else:  # Average
                if 'Category' in filtered_df.columns:
                    results = filtered_df.groupby('Category')[metric].mean().sort_values(ascending=False).head(limit)
                    
                    st.subheader(f"Average {metric} by Category")
                    
                    # Clean category names
                    clean_results = pd.Series(
                        results.values,
                        index=[self.clean_text_for_display(cat) for cat in results.index]
                    )
                    
                    st.dataframe(
                        clean_results.to_frame(f'Avg {metric}').style.format({f'Avg {metric}': '{:.2f}'}),
                        use_container_width=True
                    )
                    return
                else:
                    st.error("Category data required for average operation")
                    return
            
            # Display results
            st.subheader(f"{operation} {limit} Apps by {metric}")
            
            display_columns = ['App', metric]
            if 'Category' in results.columns:
                display_columns.insert(1, 'Category')
            if 'Rating' in results.columns and metric != 'Rating':
                display_columns.insert(-1, 'Rating')
            if 'Reviews' in results.columns and metric != 'Reviews':
                display_columns.insert(-1, 'Reviews')
            if 'Store' in results.columns:
                display_columns.append('Store')
            
            # Clean text for display
            display_df = results[display_columns].copy()
            display_df['App'] = display_df['App'].apply(self.clean_text_for_display)
            if 'Category' in display_df.columns:
                display_df['Category'] = display_df['Category'].apply(self.clean_text_for_display)
            
            # Format based on metric type
            format_dict = {}
            if 'Rating' in display_columns:
                format_dict['Rating'] = '{:.1f}'
            if 'Reviews' in display_columns:
                format_dict['Reviews'] = '{:,.0f}'
            if 'Cross_Platform_Score' in display_columns:
                format_dict['Cross_Platform_Score'] = '{:.1f}'
            
            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True
            )
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Results Found", f"{len(results)}")
            with col2:
                st.metric(f"Avg {metric}", f"{results[metric].mean():.2f}")
            with col3:
                if filters:
                    filter_text = ", ".join([f"{k}: {len(v)}" for k, v in filters.items()])
                    st.metric("Filters Applied", filter_text)
                else:
                    st.metric("Filters Applied", "None")
                    
        except Exception as e:
            st.error(f"Error executing query: {e}")
    
    def run(self):
        """Main app runner"""
        
        # App header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1>AI Market Intelligence Platform</h1>
            <p style="color: #666;">Comprehensive analysis of your actual app market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load actual data
        if not self.load_data():
            st.stop()
        
        # Show data source information in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Data Quality")
        if hasattr(self, 'df') and not self.df.empty:
            st.sidebar.success(f"**Loaded:** {len(self.df):,} apps")
            
            
            # Quality indicators
            if 'Rating' in self.df.columns:
                valid_ratings = ((self.df['Rating'] >= 1) & (self.df['Rating'] <= 5)).sum()
                st.sidebar.metric("Valid Ratings", f"{valid_ratings:,}/{len(self.df):,}")
            
            if hasattr(self, 'insights') and self.insights:
                st.sidebar.success("**AI Insights:** Available")
            else:
                st.sidebar.warning("**AI Insights:** Generate with existing code")
        
        # Render sidebar and get selected page
        selected_page = self.render_sidebar()
        
        # Render selected page
        try:
            if selected_page == "Dashboard Overview":
                self.render_dashboard_overview()
            elif selected_page == "Category Analysis":
                self.render_category_analysis()
            elif selected_page == "Store Comparison":
                self.render_store_comparison()
            elif selected_page == "Growth Opportunities":
                self.render_growth_opportunities()
            elif selected_page == "Market Intelligence":
                self.render_market_intelligence()
            elif selected_page == "Query Interface":
                self.render_query_interface()
                
        except Exception as e:
            st.error(f"Error rendering page: {e}")
            st.markdown("""
            **Troubleshooting:**
            1. Check data file integrity
            2. Restart Streamlit: `streamlit run src/interface/streamlit_app.py`
            3. Re-run pipeline: `python main.py`
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            Built with Streamlit | Data processed with comprehensive quality controls
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        app = MarketIntelligenceApp()
        app.run()
    except Exception as e:
        st.error(f"Application startup error: {e}")
        st.markdown("""
        **Setup Required:**
        1. Ensure all files are in correct directories
        2. Run: `python main.py` first
        3. Check `config/settings.py` exists
        """)
