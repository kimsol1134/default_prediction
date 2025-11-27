"""
비즈니스 가치 계산 모듈

부도 확률 기반 ROI, 기대손실, 기대수익 계산
"""

from typing import Dict


class BusinessValueCalculator:
    """비즈니스 가치 계산기"""

    def __init__(
        self,
        avg_loan: float = 5_000_000,
        avg_interest: float = 500_000,
        recovery_rate: float = 0.3
    ):
        """
        Args:
            avg_loan: 평균 대출 금액 (원)
            avg_interest: 평균 이자 수익 (원)
            recovery_rate: 부도 시 회수율 (0~1)
        """
        self.avg_loan = avg_loan
        self.avg_interest = avg_interest
        self.recovery_rate = recovery_rate

    def calculate_single_company(self, prob: float) -> Dict:
        """
        단일 기업에 대한 기대값 계산

        Args:
            prob: 부도 확률 (0~1)

        Returns:
            {
                'expected_loss': 예상 손실,
                'expected_profit': 예상 수익,
                'net': 순 기대값
            }
        """
        # 예상 손실 = 부도 확률 × 대출 금액 × (1 - 회수율)
        expected_loss = prob * self.avg_loan * (1 - self.recovery_rate)

        # 예상 수익 = (1 - 부도 확률) × 이자 수익
        expected_profit = (1 - prob) * self.avg_interest

        # 순 기대값
        net = expected_profit - expected_loss

        return {
            'expected_loss': expected_loss,
            'expected_profit': expected_profit,
            'net': net
        }

    def calculate_portfolio(self, predictions: list) -> Dict:
        """
        포트폴리오 전체에 대한 기대값 계산

        Args:
            predictions: 부도 확률 리스트

        Returns:
            포트폴리오 레벨 통계
        """
        total_loss = 0
        total_profit = 0

        for prob in predictions:
            result = self.calculate_single_company(prob)
            total_loss += result['expected_loss']
            total_profit += result['expected_profit']

        total_net = total_profit - total_loss

        return {
            'total_expected_loss': total_loss,
            'total_expected_profit': total_profit,
            'total_net': total_net,
            'num_companies': len(predictions),
            'avg_loss_per_company': total_loss / len(predictions) if predictions else 0,
            'avg_profit_per_company': total_profit / len(predictions) if predictions else 0
        }

    def get_model_performance_stats(self) -> Dict:
        """
        Part 4 노트북의 모델 성능 통계 (고정값)

        Returns:
            모델 성능 메트릭
        """
        return {
            'roi': '931%',
            'payback_months': 1.3,
            'annual_savings_krw': '466M',
            'precision': 0.0504,
            'recall': 0.8684,
            'f2_score': 0.2046
        }
