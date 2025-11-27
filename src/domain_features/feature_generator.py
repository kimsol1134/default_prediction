"""
재무제표 → 65개 도메인 특성 자동 생성

입력: 파싱된 재무제표 딕셔너리
출력: 65개 특성을 포함한 DataFrame (1행)

특성 카테고리:
1. 유동성 위기 (10개) - liquidity_features.py
2. 지급불능 (8개) - insolvency_features.py
3. 재무조작 탐지 (15개) - manipulation_features.py
4. 한국 시장 특화 (13개) - korea_market_features.py
5. 이해관계자 행동 (9개) - stakeholder_features.py
6. 복합 리스크 (7개) - composite_features.py
7. 비선형/상호작용 (3개) - composite_features.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from .liquidity_features import create_liquidity_features
from .insolvency_features import create_insolvency_features
from .manipulation_features import create_manipulation_features
from .korea_market_features import create_korea_market_features
from .stakeholder_features import create_stakeholder_features
from .composite_features import create_composite_features

logger = logging.getLogger(__name__)


class DomainFeatureGenerator:
    """도메인 기반 특성 자동 생성기"""

    def __init__(self):
        """특성 생성기 초기화"""
        self.feature_count = 0
        self.feature_categories = {
            '유동성위기': 10,
            '지급불능': 8,
            '재무조작탐지': 15,
            '한국시장특화': 13,
            '이해관계자행동': 9,
            '복합리스크': 10
        }

    def generate_all_features(
        self,
        financial_data: Dict,
        company_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        모든 도메인 특성 생성

        Args:
            financial_data: 파싱된 재무제표 데이터
                {
                    '유동자산': 1000,
                    '유동부채': 500,
                    '매출액': 3000,
                    ...
                }
            company_info: 기업 추가 정보 (선택)
                {
                    '업력': 25,
                    '외감여부': True,
                    '업종코드': 'C26',
                    '종업원수': 1000,
                    '연체여부': False,
                    '신용등급': 'A',
                    ...
                }

        Returns:
            65개 특성을 포함한 DataFrame (1행)
        """
        if company_info is None:
            company_info = {}

        all_features = {}

        try:
            # 1. 유동성 위기 특성 (10개)
            logger.info("유동성 위기 특성 생성 중...")
            liquidity = create_liquidity_features(financial_data)
            all_features.update(liquidity)
            logger.info(f"  → {len(liquidity)}개 특성 생성")

            # 2. 지급불능 특성 (8개)
            logger.info("지급불능 특성 생성 중...")
            insolvency = create_insolvency_features(financial_data)
            all_features.update(insolvency)
            logger.info(f"  → {len(insolvency)}개 특성 생성")

            # 3. 재무조작 탐지 특성 (15개)
            logger.info("재무조작 탐지 특성 생성 중...")
            manipulation = create_manipulation_features(financial_data)
            all_features.update(manipulation)
            logger.info(f"  → {len(manipulation)}개 특성 생성")

            # 4. 한국 시장 특화 특성 (13개)
            logger.info("한국 시장 특화 특성 생성 중...")
            korea_market = create_korea_market_features(financial_data, company_info)
            all_features.update(korea_market)
            logger.info(f"  → {len(korea_market)}개 특성 생성")

            # 5. 이해관계자 행동 특성 (9개)
            logger.info("이해관계자 행동 특성 생성 중...")
            stakeholder = create_stakeholder_features(financial_data, company_info)
            all_features.update(stakeholder)
            logger.info(f"  → {len(stakeholder)}개 특성 생성")

            # 6. 복합 리스크 특성 (10개) - 이전 특성들을 입력으로 사용
            logger.info("복합 리스크 특성 생성 중...")
            composite = create_composite_features(all_features)
            all_features.update(composite)
            logger.info(f"  → {len(composite)}개 특성 생성")

            # DataFrame 변환 및 검증
            df = pd.DataFrame([all_features])
            df = self._validate_and_clean(df)

            self.feature_count = len(df.columns)
            logger.info(f"✓ 총 {self.feature_count}개 특성 생성 완료")

            return df

        except Exception as e:
            logger.error(f"특성 생성 실패: {str(e)}")
            raise

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        특성 검증 및 정제

        - 무한대 값 → 0 변환
        - 결측치 → 0 변환
        - 이상치 클리핑 (예: 비율은 -10 ~ 10 범위)

        Args:
            df: 생성된 특성 DataFrame

        Returns:
            정제된 DataFrame
        """
        logger.info("특성 검증 및 정제 중...")

        original_shape = df.shape

        # 무한대 처리
        df = df.replace([np.inf, -np.inf], 0)

        # 결측치 처리
        df = df.fillna(0)

        # 이상치 클리핑
        for col in df.columns:
            # 비율, 배율 등은 -10 ~ 10 범위로 클리핑
            if any(keyword in col for keyword in ['비율', '배율', '률', '율']):
                df[col] = df[col].clip(-10, 10)

            # 일수는 0 ~ 365 범위
            if '일수' in col:
                df[col] = df[col].clip(0, 365)

            # 지수, 점수는 0 ~ 1 또는 0 ~ 100 범위
            if '지수' in col or '점수' in col:
                if df[col].max() <= 1:
                    df[col] = df[col].clip(0, 1)
                else:
                    df[col] = df[col].clip(0, 100)

        # 데이터 타입 확인
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    logger.warning(f"컬럼 '{col}'을 숫자로 변환할 수 없습니다.")

        logger.info(f"  정제 전: {original_shape}, 정제 후: {df.shape}")
        logger.info(f"  무한대/NaN 제거 완료")

        return df

    def get_feature_importance_estimates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        특성별 중요도 추정 (경험적 규칙 기반)

        Returns:
            특성명, 카테고리, 추정중요도를 포함한 DataFrame
        """
        # 경험적으로 중요한 특성들에 높은 가중치 부여
        importance_map = {
            # 유동성 (높음)
            '현금소진일수': 'critical',
            '유동비율': 'high',
            '현금비율': 'high',
            '유동성위기지수': 'critical',

            # 지급불능 (높음)
            '이자보상배율': 'critical',
            '부채비율': 'high',
            '자본잠식도': 'critical',
            '지급불능위험지수': 'critical',

            # 재무조작 (중간)
            '발생액비율': 'medium',
            '이익의질': 'high',
            '재무조작위험지수': 'high',

            # 복합 리스크 (높음)
            '종합부도위험스코어': 'critical',
            '조기경보신호수': 'critical',
            '재무건전성지수': 'high',
        }

        importance_scores = {
            'critical': 10,
            'high': 7,
            'medium': 5,
            'low': 3
        }

        results = []
        for col in df.columns:
            importance_level = importance_map.get(col, 'low')
            importance_score = importance_scores[importance_level]

            # 카테고리 판정
            if any(kw in col for kw in ['유동', '현금', '운전자본']):
                category = '유동성위기'
            elif any(kw in col for kw in ['부채', '자본', '이자', '레버리지']):
                category = '지급불능'
            elif any(kw in col for kw in ['발생액', '채권', '재고', '조작', '이익의질']):
                category = '재무조작탐지'
            elif any(kw in col for kw in ['제조업', '대기업', '외감', '업력']):
                category = '한국시장특화'
            elif any(kw in col for kw in ['연체', '체납', '신용', '배당']):
                category = '이해관계자행동'
            else:
                category = '복합리스크'

            results.append({
                '특성명': col,
                '카테고리': category,
                '중요도수준': importance_level,
                '중요도점수': importance_score
            })

        return pd.DataFrame(results).sort_values('중요도점수', ascending=False)

    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        생성된 특성의 요약 통계

        Returns:
            {
                '총특성수': 65,
                '카테고리별특성수': {...},
                '통계': {...}
            }
        """
        summary = {
            '총특성수': len(df.columns),
            '카테고리별특성수': self.feature_categories.copy(),
            '통계': {
                '평균': df.mean().to_dict(),
                '표준편차': df.std().to_dict(),
                '최소값': df.min().to_dict(),
                '최대값': df.max().to_dict()
            }
        }

        return summary
