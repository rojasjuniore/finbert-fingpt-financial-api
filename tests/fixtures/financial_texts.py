"""
Comprehensive financial text fixtures for testing.
"""
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta


class FinancialTextFixtures:
    """Collection of financial text samples for testing."""
    
    @staticmethod
    def get_positive_sentiment_texts() -> List[str]:
        """Financial texts with positive sentiment."""
        return [
            # Earnings beats
            "Apple Inc. reported record quarterly earnings, beating analyst expectations by 15% with revenue of $94.9 billion.",
            "Microsoft announces strong cloud revenue growth of 32% and raises guidance for next quarter.",
            "Tesla delivers record quarterly vehicle deliveries, exceeding Wall Street estimates by 18%.",
            "Amazon Web Services shows continued momentum with 28% growth in quarterly revenue.",
            "NVIDIA reports exceptional Q3 results with data center revenue up 206% year-over-year.",
            
            # Stock performance
            "Tesla stock surges 20% after announcing breakthrough in battery technology and cost reduction.",
            "Apple shares reach all-time high following strong iPhone sales and services revenue growth.",
            "Microsoft stock climbs 15% on strong Azure cloud computing demand and AI integration news.",
            "Google parent Alphabet gains 12% after beating earnings with robust advertising revenue.",
            "Amazon stock rises 18% on strong holiday shopping season and AWS performance.",
            
            # Strategic moves
            "Company announces successful acquisition of leading AI startup, enhancing competitive position.",
            "Strategic partnership with Fortune 500 company expected to drive $500M in annual revenue.",
            "New product launch receives overwhelmingly positive reviews from industry analysts.",
            "Company secures major government contract worth $1.2 billion over five years.",
            "International expansion into European markets shows strong early adoption rates.",
            
            # Financial health
            "Company reports strong balance sheet with $15 billion in cash and minimal debt levels.",
            "Dividend increase of 12% announced, marking 25th consecutive year of dividend growth.",
            "Share buyback program expanded by $10 billion, demonstrating confidence in future prospects.",
            "Free cash flow generation increases 35% year-over-year, exceeding management guidance.",
            "Credit rating upgraded to AAA reflecting strong financial position and cash generation.",
            
            # Market position
            "Market share gains accelerate with 25% increase in customer base over past quarter.",
            "Brand value rises to top 5 globally according to latest Interbrand rankings.",
            "Customer satisfaction scores reach all-time highs across all product categories.",
            "Innovation pipeline strengthened with 200+ patents filed in past year.",
            "Supply chain resilience initiatives result in 40% reduction in delivery times."
        ]
    
    @staticmethod
    def get_negative_sentiment_texts() -> List[str]:
        """Financial texts with negative sentiment."""
        return [
            # Earnings misses
            "Company reports disappointing quarterly results, missing analyst estimates by significant margin.",
            "Revenue declines 15% year-over-year as market conditions remain challenging.",
            "Quarterly losses exceed expectations, causing concern among institutional investors.",
            "Profit margins compressed due to rising costs and competitive pricing pressure.",
            "Guidance lowered for full year as management cites economic headwinds.",
            
            # Stock declines
            "Stock price plummets 25% following disappointing earnings announcement and weak guidance.",
            "Shares hit 52-week low as investors react to management's cautious outlook.",
            "Market cap erodes by $20 billion amid concerns about competitive positioning.",
            "Stock underperforms sector by 30% year-to-date as growth concerns mount.",
            "Share price volatility increases as uncertainty about future prospects grows.",
            
            # Operational challenges
            "Supply chain disruptions continue to impact production and delivery capabilities.",
            "Regulatory investigation threatens future business operations and profitability.",
            "Key executive departures raise questions about strategic direction and leadership.",
            "Product recall costs expected to exceed $500 million in current quarter.",
            "Cybersecurity breach exposes customer data, prompting regulatory scrutiny.",
            
            # Financial distress
            "Credit rating downgraded due to mounting debt levels and declining cash flow.",
            "Liquidity concerns emerge as company burns through cash reserves rapidly.",
            "Debt-to-equity ratio reaches concerning levels, limiting financial flexibility.",
            "Interest coverage ratio falls below covenant requirements, triggering lender discussions.",
            "Working capital challenges strain vendor relationships and operational efficiency.",
            
            # Market challenges
            "Market share erosion accelerates as competitors gain ground with innovative products.",
            "Customer churn rates increase significantly, impacting recurring revenue streams.",
            "Pricing power diminishes as commoditization trends accelerate in core markets.",
            "International operations face headwinds from currency devaluation and political instability.",
            "Technology disruption threatens traditional business model sustainability."
        ]
    
    @staticmethod
    def get_neutral_sentiment_texts() -> List[str]:
        """Financial texts with neutral sentiment."""
        return [
            # Routine announcements
            "The company will hold its annual shareholders meeting next month in Delaware.",
            "Board of directors announces retirement of long-serving CEO effective year-end.",
            "Company files routine quarterly regulatory documents with the Securities and Exchange Commission.",
            "Annual report published with detailed financial statements and business overview.",
            "Quarterly dividend payment date announced for shareholders of record.",
            
            # Corporate governance
            "New independent director appointed to board bringing technology industry expertise.",
            "Audit committee reviews internal controls and risk management procedures.",
            "Corporate governance policies updated to reflect best practices and regulatory changes.",
            "Compensation committee conducts annual review of executive pay structures.",
            "Stockholder proposals for annual meeting published in proxy statement.",
            
            # Operational updates
            "Company completes routine facility maintenance shutdown lasting two weeks.",
            "Employee training programs expanded to include new safety and compliance protocols.",
            "Information technology systems upgrade scheduled for completion by year-end.",
            "Organizational restructuring aims to streamline operations and improve efficiency.",
            "New manufacturing facility construction remains on schedule for Q3 opening.",
            
            # Industry participation
            "Company participates in industry trade show showcasing latest product innovations.",
            "Management team attends investor conference to discuss business strategy.",
            "Sustainability report details environmental and social responsibility initiatives.",
            "Industry association membership renewed with continued board participation.",
            "Research and development collaboration with university announced.",
            
            # Factual statements
            "Company operates manufacturing facilities in 15 countries worldwide.",
            "Workforce totals approximately 50,000 employees across global operations.",
            "Product portfolio includes over 200 distinct items serving diverse markets.",
            "Distribution network encompasses 10,000 retail locations in North America.",
            "Founded in 1985, company has maintained consistent growth trajectory."
        ]
    
    @staticmethod
    def get_mixed_sentiment_texts() -> List[str]:
        """Financial texts with mixed or complex sentiment."""
        return [
            # Mixed results
            "Despite strong revenue growth of 20%, the company reported lower profit margins due to increased operational costs.",
            "While domestic sales exceeded expectations, international performance remained challenged by currency headwinds.",
            "Successful product launch offset by higher than anticipated marketing and promotional expenses.",
            "Strong cash generation enabled debt reduction, though credit rating remains unchanged pending further improvement.",
            "Market share gains in core business balanced by losses in emerging product categories.",
            
            # Transitional periods
            "Acquisition integration proceeding smoothly, though expected synergies have not yet materialized.",
            "Digital transformation initiatives show promise but have not yet translated to revenue gains.",
            "Cost reduction program delivered savings, however implementation resulted in temporary operational disruptions.",
            "New management team brings fresh perspective while maintaining strategic continuity with previous leadership.",
            "Restructuring charges impacted current quarter results but are expected to improve future profitability.",
            
            # Balanced outlooks
            "Cautiously optimistic outlook for next year balances growth opportunities with economic uncertainties.",
            "Innovation investments positioned for long-term success despite near-term margin pressure.",
            "Geographic diversification provides stability while exposing company to varied regional risks.",
            "Strong brand recognition supports pricing power, though competitive pressures remain intense.",
            "Regulatory compliance costs increase operational expenses but strengthen competitive moat."
        ]
    
    @staticmethod
    def get_edge_case_texts() -> List[str]:
        """Edge cases and challenging texts for testing."""
        return [
            # Empty and minimal
            "",
            " ",
            "A",
            "No.",
            "Q3 2024.",
            
            # Very short financial terms
            "Bull market",
            "Bear market",
            "Stock up",
            "Earnings miss",
            "Revenue growth",
            
            # Very long text
            "The comprehensive financial analysis " * 100 + "shows mixed results across all business segments.",
            
            # Special characters
            "Company's Q3 earnings: +15% YoY! 📈 $AAPL #bullish @investors",
            "Revenue: $1,234.56M (+12.3% QoQ) - STRONG 💪",
            "🚀 Stock price ⬆️ 25% after earnings beat! 💰💰💰",
            
            # Unicode and international
            "Société reported strong résultats financiers with revenue of €1.2 billion",
            "東京証券取引所での株価は15%上昇しました",
            "Прибыль компании выросла на 20% в третьем квартале",
            
            # Numbers and symbols
            "Revenue: $1,234,567,890 (+15.7% YoY) vs Est. $1,200,000,000",
            "EPS: $2.45 (vs $2.30 est.) | Rev: $15.2B (vs $14.8B est.)",
            "P/E: 23.4x | P/B: 3.2x | ROE: 18.5% | Debt/Equity: 0.42x",
            
            # Technical jargon
            "EBITDA margins expanded 150bps QoQ to 23.4% driven by operational leverage",
            "FCF generation of $2.3B represents 8.2% FCF yield on current market cap",
            "ROIC improved 200bps to 15.2% as NOPAT growth outpaced invested capital",
            
            # Ambiguous context
            "Results were as expected given current market conditions and competitive dynamics",
            "Performance aligned with management guidance provided during previous quarter",
            "Outcome consistent with industry trends and macroeconomic environment",
            
            # Contradictory statements
            "Strong revenue growth masked underlying profitability challenges",
            "Positive headline numbers concealed concerning operational metrics",
            "Success in new markets offset by deterioration in traditional business",
            
            # Hypothetical/conditional
            "If market conditions improve, the company could see significant upside potential",
            "Should regulatory approval be granted, revenue impact could be substantial",
            "Assuming successful integration, synergies may exceed original estimates"
        ]
    
    @staticmethod
    def get_sector_specific_texts() -> Dict[str, List[str]]:
        """Sector-specific financial texts."""
        return {
            "technology": [
                "Cloud computing revenue accelerated 45% year-over-year driven by enterprise adoption",
                "AI and machine learning capabilities drive customer engagement and retention",
                "Semiconductor shortage impacts hardware production and delivery timelines",
                "Software-as-a-Service recurring revenue model provides predictable cash flows",
                "Cybersecurity solutions see increased demand amid rising threat landscape"
            ],
            
            "healthcare": [
                "Drug approval by FDA opens $5 billion addressable market opportunity",
                "Clinical trial results exceed efficacy endpoints with strong safety profile",
                "Healthcare services segment benefits from aging demographic trends",
                "Medical device innovation drives market share gains in cardiology segment",
                "Regulatory delays impact new product launch timeline and revenue recognition"
            ],
            
            "finance": [
                "Net interest margin expansion supports strong quarterly earnings growth",
                "Credit loss provisions normalized following pandemic-related increases",
                "Investment banking fees decline amid reduced M&A and IPO activity",
                "Digital banking transformation reduces operational costs and improves efficiency",
                "Regulatory capital ratios remain well above minimum requirements"
            ],
            
            "energy": [
                "Oil prices recovery supports upstream exploration and production activities",
                "Renewable energy investments position company for energy transition",
                "Natural gas production increases meeting growing demand for cleaner fuels",
                "Refining margins compressed due to inventory adjustments and demand patterns",
                "Carbon capture technology investments align with environmental regulations"
            ],
            
            "retail": [
                "E-commerce sales growth accelerates with improved fulfillment capabilities",
                "Same-store sales increase driven by improved customer experience initiatives",
                "Supply chain optimization reduces inventory levels and improves working capital",
                "Private label products drive margin expansion and customer loyalty",
                "Store footprint optimization balances physical presence with digital strategy"
            ]
        }
    
    @staticmethod
    def get_expected_sentiment_mappings() -> Dict[str, str]:
        """Expected sentiment labels for specific test texts."""
        return {
            "Apple Inc. reported record quarterly earnings, beating analyst expectations by 15%.": "positive",
            "Company faces significant challenges due to supply chain disruptions.": "negative",
            "The company will hold its annual shareholders meeting next month.": "neutral",
            "Tesla stock surges 20% after announcing breakthrough in battery technology.": "positive",
            "Quarterly losses exceed expectations, causing concern among investors.": "negative",
            "Board of directors announces retirement of long-serving CEO.": "neutral",
            "Strong revenue growth": "positive",
            "Stock price plummets": "negative",
            "Routine announcement": "neutral"
        }
    
    @staticmethod
    def get_performance_test_texts() -> List[str]:
        """Optimized texts for performance testing."""
        return [
            # Short texts (fast processing)
            "Strong earnings beat",
            "Revenue declined",
            "Market neutral",
            "Positive guidance",
            "Profit warning",
            
            # Medium texts (normal processing)
            "Company reports strong quarterly results exceeding analyst expectations",
            "Market conditions remain challenging impacting near-term performance outlook",
            "Strategic initiatives progress according to plan with measured results",
            
            # Longer texts (slower processing)
            "Comprehensive quarterly analysis reveals mixed performance across business segments with revenue growth offset by margin compression",
            "Management commentary during earnings call highlighted both opportunities and challenges facing the organization in current market environment"
        ]
    
    @staticmethod
    def get_batch_test_scenarios() -> List[Dict[str, Any]]:
        """Batch processing test scenarios."""
        return [
            {
                "name": "mixed_sentiment_batch",
                "texts": [
                    "Strong quarterly earnings beat expectations",
                    "Revenue declined due to market headwinds", 
                    "Company announces routine board meeting"
                ],
                "expected_labels": ["positive", "negative", "neutral"]
            },
            {
                "name": "large_uniform_batch",
                "texts": ["Positive earnings surprise"] * 20,
                "expected_labels": ["positive"] * 20
            },
            {
                "name": "varying_length_batch",
                "texts": [
                    "Beat",
                    "Company reports strong quarterly performance",
                    "Comprehensive analysis of financial results reveals robust growth across multiple business segments with particular strength in core markets"
                ],
                "expected_labels": ["positive", "positive", "positive"]
            }
        ]
    
    @staticmethod
    def save_fixtures_to_file(filepath: str):
        """Save all fixtures to JSON file for external use."""
        fixtures = {
            "positive_texts": FinancialTextFixtures.get_positive_sentiment_texts(),
            "negative_texts": FinancialTextFixtures.get_negative_sentiment_texts(),
            "neutral_texts": FinancialTextFixtures.get_neutral_sentiment_texts(),
            "mixed_texts": FinancialTextFixtures.get_mixed_sentiment_texts(),
            "edge_cases": FinancialTextFixtures.get_edge_case_texts(),
            "sector_specific": FinancialTextFixtures.get_sector_specific_texts(),
            "expected_mappings": FinancialTextFixtures.get_expected_sentiment_mappings(),
            "performance_texts": FinancialTextFixtures.get_performance_test_texts(),
            "batch_scenarios": FinancialTextFixtures.get_batch_test_scenarios(),
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_texts": sum([
                    len(FinancialTextFixtures.get_positive_sentiment_texts()),
                    len(FinancialTextFixtures.get_negative_sentiment_texts()),
                    len(FinancialTextFixtures.get_neutral_sentiment_texts()),
                    len(FinancialTextFixtures.get_mixed_sentiment_texts()),
                    len(FinancialTextFixtures.get_edge_case_texts())
                ]),
                "categories": [
                    "positive", "negative", "neutral", "mixed", 
                    "edge_cases", "sector_specific", "performance"
                ]
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(fixtures, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    fixtures = FinancialTextFixtures()
    
    print("Positive texts:", len(fixtures.get_positive_sentiment_texts()))
    print("Negative texts:", len(fixtures.get_negative_sentiment_texts()))
    print("Neutral texts:", len(fixtures.get_neutral_sentiment_texts()))
    print("Mixed texts:", len(fixtures.get_mixed_sentiment_texts()))
    print("Edge cases:", len(fixtures.get_edge_case_texts()))
    
    # Save to file
    fixtures.save_fixtures_to_file("financial_text_fixtures.json")