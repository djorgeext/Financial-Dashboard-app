"""
Options analysis service for generating options trading recommendations.
Calculates option strategies based on forecast direction and probability.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Generates options trading recommendations based on forecasts."""
    
    # Strike selection offsets (as % of current price)
    CALL_STRIKES = [0.01, 0.02, 0.03]  # 1%, 2%, 3% OTM
    PUT_STRIKES = [-0.01, -0.02, -0.03]  # 1%, 2%, 3% OTM
    
    # Expiration dates
    EXPIRATIONS = [7, 14, 21]  # days
    
    def suggest_options(
        self,
        ticker: str,
        current_price: float,
        forecast_price: float,
        forecast_confidence: float,
        pred_class: int = 1  # 0=down, 1=neutral, 2=up
    ) -> Dict:
        """
        Generate options trading suggestions based on forecast.
        
        Args:
            ticker: Stock ticker
            current_price: Current market price
            forecast_price: Predicted price from model
            forecast_confidence: Model confidence (0-1)
            pred_class: Direction prediction (0=down, 1=neutral, 2=up)
            
        Returns:
            Dict with suggested options
        """
        try:
            # Determine direction
            price_move = (forecast_price - current_price) / current_price
            
            suggestions = []
            
            if pred_class == 2 or price_move > 0.005:  # Bullish
                suggestions.extend(self._generate_call_suggestions(
                    ticker, current_price, forecast_confidence
                ))
            
            if pred_class == 0 or price_move < -0.005:  # Bearish
                suggestions.extend(self._generate_put_suggestions(
                    ticker, current_price, forecast_confidence
                ))
            
            if pred_class == 1 and abs(price_move) < 0.005:  # Neutral
                suggestions.extend(self._generate_strangle_suggestions(
                    ticker, current_price, forecast_confidence
                ))
            
            # Sort by probability descending
            suggestions.sort(key=lambda x: x["probability"], reverse=True)
            
            return {
                "ticker": ticker,
                "current_price": float(current_price),
                "forecast_price": float(forecast_price),
                "forecast_direction": self._get_direction_label(pred_class),
                "suggested_options": suggestions[:5],  # Top 5 suggestions
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating options suggestions: {e}")
            return {
                "ticker": ticker,
                "current_price": float(current_price),
                "suggested_options": [],
                "error": str(e)
            }
    
    def _generate_call_suggestions(
        self,
        ticker: str,
        current_price: float,
        confidence: float
    ) -> List[Dict]:
        """Generate call option suggestions for bullish outlook."""
        suggestions = []
        
        for i, strike_offset in enumerate(self.CALL_STRIKES):
            strike = current_price * (1 + strike_offset)
            
            for exp_days in self.EXPIRATIONS:
                exp_date = datetime.now() + timedelta(days=exp_days)
                
                # Calculate probability using confidence and distance
                prob = confidence * (1 - strike_offset * 0.5)  # Closer strikes have higher prob
                prob = max(0.3, min(0.95, prob))  # Clamp between 0.3 and 0.95
                
                # ATM premium approximation (simplified Black-Scholes)
                days_to_exp = exp_days
                volatility = 0.25  # ~25% annualized volatility
                time_to_exp = days_to_exp / 365.0
                
                premium = self._estimate_option_premium(
                    current_price, strike, volatility, time_to_exp,
                    option_type="call"
                )
                
                suggestions.append({
                    "option_type": "call",
                    "strike": float(round(strike, 2)),
                    "expiration": exp_date.strftime("%Y-%m-%d"),
                    "recommendation": "Buy" if prob > 0.65 else "Consider",
                    "probability": float(round(prob, 2)),
                    "premium": float(round(premium, 2)),
                    "rationale": f"Bullish forecast. Strike {strike_offset*100:.0f}% OTM provides {exp_days}d exposure",
                    "time_value": float(round(premium * 0.4, 2))  # Approximate time value
                })
        
        return suggestions
    
    def _generate_put_suggestions(
        self,
        ticker: str,
        current_price: float,
        confidence: float
    ) -> List[Dict]:
        """Generate put option suggestions for bearish outlook."""
        suggestions = []
        
        for i, strike_offset in enumerate(self.PUT_STRIKES):
            strike = current_price * (1 + strike_offset)  # strike_offset is negative
            
            for exp_days in self.EXPIRATIONS:
                exp_date = datetime.now() + timedelta(days=exp_days)
                
                # Calculate probability
                prob = confidence * (1 - abs(strike_offset) * 0.5)
                prob = max(0.3, min(0.95, prob))
                
                volatility = 0.25
                time_to_exp = exp_days / 365.0
                
                premium = self._estimate_option_premium(
                    current_price, strike, volatility, time_to_exp,
                    option_type="put"
                )
                
                suggestions.append({
                    "option_type": "put",
                    "strike": float(round(strike, 2)),
                    "expiration": exp_date.strftime("%Y-%m-%d"),
                    "recommendation": "Buy" if prob > 0.65 else "Protective",
                    "probability": float(round(prob, 2)),
                    "premium": float(round(premium, 2)),
                    "rationale": f"Bearish forecast. Strike {abs(strike_offset)*100:.0f}% OTM for downside protection",
                    "time_value": float(round(premium * 0.4, 2))
                })
        
        return suggestions
    
    def _generate_strangle_suggestions(
        self,
        ticker: str,
        current_price: float,
        confidence: float
    ) -> List[Dict]:
        """Generate strangle (volatility) suggestions for neutral outlook."""
        suggestions = []
        
        call_strike = current_price * 1.02
        put_strike = current_price * 0.98
        
        for exp_days in self.EXPIRATIONS:
            exp_date = datetime.now() + timedelta(days=exp_days)
            
            # Lower probability for neutral/volatility plays
            prob = confidence * 0.6
            prob = max(0.3, min(0.9, prob))
            
            volatility = 0.25
            time_to_exp = exp_days / 365.0
            
            call_premium = self._estimate_option_premium(
                current_price, call_strike, volatility, time_to_exp,
                option_type="call"
            )
            put_premium = self._estimate_option_premium(
                current_price, put_strike, volatility, time_to_exp,
                option_type="put"
            )
            
            total_cost = call_premium + put_premium
            
            suggestions.append({
                "option_type": "strangle",
                "call_strike": float(round(call_strike, 2)),
                "put_strike": float(round(put_strike, 2)),
                "expiration": exp_date.strftime("%Y-%m-%d"),
                "recommendation": "Volatility Play",
                "probability": float(round(prob, 2)),
                "total_cost": float(round(total_cost, 2)),
                "rationale": f"Neutral to volatile. Long {exp_days}d strangle benefits from IV expansion",
                "breakevens": [
                    float(round(put_strike - total_cost, 2)),
                    float(round(call_strike + total_cost, 2))
                ]
            })
        
        return suggestions
    
    def _estimate_option_premium(
        self,
        spot: float,
        strike: float,
        volatility: float,
        time_to_exp: float,
        option_type: str = "call",
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Estimate option premium using simplified Black-Scholes.
        
        Args:
            spot: Current stock price
            strike: Strike price
            volatility: Annualized volatility
            time_to_exp: Time to expiration in years
            option_type: "call" or "put"
            risk_free_rate: Risk-free rate (5% default)
            
        Returns:
            Estimated premium
        """
        from scipy.stats import norm
        
        if spot <= 0 or strike <= 0 or volatility <= 0 or time_to_exp <= 0:
            return 0.0
        
        try:
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_exp) / (
                volatility * np.sqrt(time_to_exp)
            )
            d2 = d1 - volatility * np.sqrt(time_to_exp)
            
            if option_type == "call":
                premium = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_exp) * norm.cdf(d2)
            else:  # put
                premium = strike * np.exp(-risk_free_rate * time_to_exp) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            
            return float(max(0.01, premium))  # Minimum 1 cent
        
        except (ValueError, OverflowError):
            # Fallback to intrinsic value only
            if option_type == "call":
                return float(max(0.0, spot - strike))
            else:
                return float(max(0.0, strike - spot))
    
    def _get_direction_label(self, pred_class: int) -> str:
        """Convert pred_class to direction label."""
        if pred_class == 0:
            return "bearish"
        elif pred_class == 2:
            return "bullish"
        else:
            return "neutral"


def create_options_analyzer() -> OptionsAnalyzer:
    """Factory function to create options analyzer."""
    return OptionsAnalyzer()
