"""
한국 시장 특화 특성 생성 (13개)

한국 시장 특성:
- 제조업 중심 경제
- 대기업 의존도
- 외감 여부
- 업력
- 등등

주의: 일부 특성은 DART API에서 제공하지 않으므로
      company_info에서 제공하거나 기본값 사용
"""

from typing import Dict, Optional
import numpy as np


def create_korea_market_features(
    financial_data: Dict,
    company_info: Optional[Dict] = None
) -> Dict:
    """
    한국 시장 특화 특성 13개 생성

    Args:
        financial_data: 파싱된 재무제표 딕셔너리
        company_info: 기업 추가 정보 (선택)
            {
                '업력': 25,
                '외감여부': True,
                '업종코드': 'C26',
                '종업원수': 1000,
                ...
            }

    Returns:
        {
            '제조업여부': 1.0,
            '대기업여부': 1.0,
            '외감여부': 1.0,
            ...
        }

    특성 설명:
    - 제조업여부: 제조업이면 1, 아니면 0
    - 대기업여부: 자산 5조원 이상이면 1
    - 외감여부: 외부감사 대상이면 1
    - 업력: 설립 후 경과 년수
    - 종업원수: 직원 수
    """
    features = {}

    # DART API에서 일부 제공 (업종코드), 나머지는 기본값 사용
    # DART company.json API로 업종코드, 설립일 등 추가 가능
    if company_info is None:
        company_info = {}

    # 데이터 추출
    자산총계 = financial_data.get('자산총계', 0)
    매출액 = financial_data.get('매출액', 0)
    유형자산 = financial_data.get('유형자산', 0)
    재고자산 = financial_data.get('재고자산', 0)
    매출원가 = financial_data.get('매출원가', 0)

    # 1. 제조업여부 (업종코드 또는 유형자산 비중으로 추정)
    # KSIC 코드 C (제조업) 또는 유형자산 비중 > 30%
    업종코드 = company_info.get('업종코드', '')
    if 업종코드.startswith('C'):
        features['제조업여부'] = 1.0
    else:
        # 유형자산 비중으로 추정
        유형자산비중 = 유형자산 / (자산총계 + 1)
        features['제조업여부'] = 1.0 if 유형자산비중 > 0.3 else 0.0

    # 2. 대기업여부 (자산 5조원 이상, 백만원 단위)
    features['대기업여부'] = 1.0 if 자산총계 >= 5_000_000 else 0.0

    # 3. 외감여부 (외부감사 대상)
    # 상장기업, 자산 500억 이상 등
    외감여부 = company_info.get('외감여부', 자산총계 >= 50_000)
    features['외감여부'] = 1.0 if 외감여부 else 0.0

    # 4. 업력 (설립 후 경과 년수)
    업력 = company_info.get('업력', 10)  # 기본값 10년
    features['업력'] = float(업력)

    # 5. 신생기업여부 (업력 3년 미만)
    features['신생기업여부'] = 1.0 if 업력 < 3 else 0.0

    # 6. 종업원수 (로그 스케일)
    종업원수 = company_info.get('종업원수', 10000)  # 상장사 기본값
    features['종업원수_로그'] = np.log1p(종업원수)

    # 7. 인당매출액 = 매출액 / 종업원수
    features['인당매출액'] = 매출액 / (종업원수 + 1)

    # 8. 인당자산 = 자산 / 종업원수
    features['인당자산'] = 자산총계 / (종업원수 + 1)

    # 9. 자본집약도 = 유형자산 / 종업원수
    # 제조업의 경우 높음
    features['자본집약도'] = 유형자산 / (종업원수 + 1)

    # 10. 제조원가율 = 매출원가 / 매출액
    # 제조업의 경우 높음
    features['제조원가율'] = 매출원가 / (매출액 + 1)

    # 11. 재고집약도 = 재고자산 / 자산
    # 제조업/유통업의 경우 높음
    features['재고집약도'] = 재고자산 / (자산총계 + 1)

    # 12. 설비투자비중 = 유형자산 / 자산
    features['설비투자비중'] = 유형자산 / (자산총계 + 1)

    # 13. 한국시장리스크지수 (복합 지표)
    # 제조업 + 신생기업 + 외감 없음 = 리스크 높음
    제조업리스크 = features['제조업여부'] * 0.3  # 제조업은 경기 민감
    신생기업리스크 = features['신생기업여부'] * 0.4  # 신생기업은 부도율 높음
    외감리스크 = (1 - features['외감여부']) * 0.3  # 외감 없으면 투명성 낮음

    features['한국시장리스크지수'] = (
        제조업리스크 + 신생기업리스크 + 외감리스크
    )

    # 10. 매출집중도 (자산 대비 매출 비중)
    # 매출액 / 자산총계
    features['매출집중도'] = 매출액 / (자산총계 + 1)

    # 11. 매출집중리스크 (매출집중도가 너무 높거나 낮으면 리스크)
    # 0.5 ~ 1.5 사이가 적정하다고 가정
    매출집중도 = features['매출집중도']
    if 매출집중도 < 0.5:
        features['매출집중리스크'] = 1
    elif 매출집중도 > 1.5:
        features['매출집중리스크'] = 2
    else:
        features['매출집중리스크'] = 0

    # 무한대/NaN 처리
    for key in features:
        if not np.isfinite(features[key]):
            features[key] = 0.0

    return features
