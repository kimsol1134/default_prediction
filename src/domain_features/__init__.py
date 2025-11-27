"""
도메인 기반 특성 공학 모듈

재무제표 데이터를 입력받아 65개 도메인 특성을 자동 생성합니다.

특성 카테고리:
1. 유동성 위기 (10개) - liquidity_features
2. 지급불능 (8개) - insolvency_features
3. 재무조작 탐지 (15개) - manipulation_features
4. 한국 시장 특화 (13개) - korea_market_features
5. 이해관계자 행동 (9개) - stakeholder_features
6. 복합 리스크 (7개) - composite_features
7. 비선형/상호작용 (3개) - composite_features
"""

from .feature_generator import DomainFeatureGenerator

__all__ = ['DomainFeatureGenerator']
