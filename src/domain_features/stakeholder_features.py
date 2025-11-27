"""
이해관계자 행동 특성 생성 (9개)

이해관계자 행동:
- 연체 여부
- 세금 체납
- 신용등급
- 대표이사 변경
- 등등

주의: DART API에서 제공하지 않는 정보가 많으므로
      company_info에서 제공하거나 재무제표에서 추정
"""

from typing import Dict, Optional
import numpy as np


def create_stakeholder_features(
    financial_data: Dict,
    company_info: Optional[Dict] = None
) -> Dict:
    """
    이해관계자 행동 특성 9개 생성

    Args:
        financial_data: 파싱된 재무제표 딕셔너리
        company_info: 기업 추가 정보 (선택)
            {
                '연체여부': False,
                '세금체납액': 0,
                '신용등급': 'A',
                '대표이사변경': False,
                ...
            }

    Returns:
        {
            '연체여부': 0.0,
            '세금체납여부': 0.0,
            '신용등급점수': 7.0,
            ...
        }

    특성 설명:
    - 연체여부: 대출/어음 연체 시 1
    - 세금체납여부: 세금 체납 시 1
    - 신용등급점수: AAA=10, AA=9, ..., D=1
    - 이자지급부담도: 이자비용/영업이익
    """
    features = {}

    # DART API에서 제공하지 않는 정보는 안전한 기본값 사용
    # (신용평가사 데이터, 나이스평가 등 외부 데이터 필요)
    if company_info is None:
        company_info = {}

    # 데이터 추출
    영업이익 = financial_data.get('영업이익', 0)
    이자비용 = financial_data.get('이자비용', 0)
    당기순이익 = financial_data.get('당기순이익', 0)
    자본총계 = financial_data.get('자본총계', 0)
    부채총계 = financial_data.get('부채총계', 0)
    영업활동현금흐름 = financial_data.get('영업활동현금흐름', 0)

    # 1. 연체여부
    연체여부 = company_info.get('연체여부', False)
    features['연체여부'] = 1.0 if 연체여부 else 0.0

    # 2. 세금체납여부
    세금체납액 = company_info.get('세금체납액', 0)
    features['세금체납여부'] = 1.0 if 세금체납액 > 0 else 0.0

    # 3. 신용등급점수 (AAA=10, AA=9, A=8, BBB=7, BB=6, B=5, CCC=4, CC=3, C=2, D=1)
    신용등급 = company_info.get('신용등급', 'BBB')
    등급점수_매핑 = {
        'AAA': 10, 'AA': 9, 'A': 8, 'BBB': 7, 'BB': 6,
        'B': 5, 'CCC': 4, 'CC': 3, 'C': 2, 'D': 1
    }
    features['신용등급점수'] = 등급점수_매핑.get(신용등급.upper(), 7)

    # 4. 대표이사변경여부 (최근 1년 내)
    대표이사변경 = company_info.get('대표이사변경', False)
    features['대표이사변경여부'] = 1.0 if 대표이사변경 else 0.0

    # 5. 이자지급부담도 = 이자비용 / 영업이익
    # 높을수록 이자 부담 큼
    features['이자지급부담도'] = 이자비용 / (영업이익 + 1)

    # 6. 배당여부 (당기순이익이 양수이고 배당 지급)
    배당금 = company_info.get('배당금', 0)
    features['배당여부'] = 1.0 if 배당금 > 0 and 당기순이익 > 0 else 0.0

    # 7. 배당성향 = 배당금 / 당기순이익
    features['배당성향'] = 배당금 / (당기순이익 + 1)

    # 8. 부실징후종합점수 (0~1, 높을수록 부실 가능성 높음)
    # = 연체 + 체납 + (10 - 신용등급점수)/10 + 대표이사변경*0.2
    부실점수 = (
        features['연체여부'] * 0.4 +
        features['세금체납여부'] * 0.3 +
        (10 - features['신용등급점수']) / 10 * 0.2 +
        features['대표이사변경여부'] * 0.1
    )
    features['부실징후종합점수'] = min(1.0, 부실점수)

    # 4. 이해관계자불신지수 (복합 지표)
    # = 연체 + 체납 + 낮은 신용등급 + 배당 안 함
    # 여기서는 단순화하여 연체여부, 세금체납여부, 법적리스크(없으면 0) 사용
    불신지수 = (
        0.4 * features['연체여부'] +
        0.3 * features['세금체납여부'] +
        0.3 * features.get('법적리스크', 0.0)
    )
    features['이해관계자불신지수'] = min(1.0, 불신지수)

    # 5. 연체심각도 = 총연체건수 * (부채비율/100)
    # 총연체건수가 features에 없으므로 company_info에서 가져오거나 계산
    features['총연체건수'] = company_info.get('총연체건수', 0)
    
    # 부채비율은 여기서 계산 불가하므로, 외부에서 주입받거나 여기서 근사 계산
    # financial_data에 부채총계, 자본총계가 있으므로 계산 가능
    부채총계 = financial_data.get('부채총계', 0)
    자본총계 = financial_data.get('자본총계', 0)
    부채비율 = (부채총계 / (자본총계 + 1)) * 100
    features['연체심각도'] = features['총연체건수'] * (부채비율 / 100)

    # 6. 공공정보리스크 (세금체납 등 공공 정보 기반 리스크)
    # 여기서는 세금체납여부만 있으므로 이를 기반으로 생성
    features['공공정보리스크'] = features['세금체납여부'] * 3  # 가중치 3

    # 7. 이해관계자_불신지수 (Part3에서 제거됨 - 논리적 오류)
    # Part3 노트북에서 '이해관계자불신지수'와 중복으로 제거
    # features['이해관계자_불신지수'] = (features['연체심각도'] + features['공공정보리스크']) / 2

    # 무한대/NaN 처리
    for key in features:
        if not np.isfinite(features[key]):
            features[key] = 0.0

    return features
