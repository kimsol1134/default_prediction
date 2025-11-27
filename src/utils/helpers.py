"""
헬퍼 함수들

리스크 등급 판정, 숫자 포맷팅 등
"""

from typing import Tuple, Dict, List
import numpy as np


def get_risk_level(risk_score: float) -> Tuple[str, str, str]:
    """
    위험도 점수를 등급으로 변환 (Part 3 최적 임계값 사용)

    Args:
        risk_score: 부도 확률 (0~1)

    Returns:
        (등급명, 이모지, 설명)
    """
    if risk_score < 0.0168:  # < 1.68%
        return ("안전", "🟢", "부도 위험이 매우 낮습니다")
    elif risk_score < 0.0468:  # < 4.68%
        return ("주의", "🟡", "잠재적 위험 요소가 있습니다")
    else:  # >= 4.68%
        return ("고위험", "🔴", "부도 위험이 높습니다")


def format_korean_number(number: float, unit: str = "원") -> str:
    """
    숫자를 한국식 단위로 포맷팅

    Args:
        number: 숫자 (백만원 단위)
        unit: 단위 문자열

    Returns:
        포맷팅된 문자열 (예: "1조 2,000억원")
    """
    if number == 0:
        return "0" + unit

    # 백만원 단위 → 원 단위
    number_won = number * 1_000_000

    if number_won >= 1_000_000_000_000:  # 1조 이상
        jo = number_won // 1_000_000_000_000
        eok = (number_won % 1_000_000_000_000) // 100_000_000
        if eok > 0:
            return f"{jo:,.0f}조 {eok:,.0f}억{unit}"
        else:
            return f"{jo:,.0f}조{unit}"
    elif number_won >= 100_000_000:  # 1억 이상
        eok = number_won // 100_000_000
        man = (number_won % 100_000_000) // 10_000
        if man > 0:
            return f"{eok:,.0f}억 {man:,.0f}만{unit}"
        else:
            return f"{eok:,.0f}억{unit}"
    elif number_won >= 10_000:  # 1만 이상
        man = number_won // 10_000
        return f"{man:,.0f}만{unit}"
    else:
        return f"{number_won:,.0f}{unit}"


def calculate_percentile(value: float, distribution: List[float]) -> float:
    """
    분포 내에서 값의 백분위수 계산

    Args:
        value: 계산할 값
        distribution: 비교 분포

    Returns:
        백분위수 (0~100)
    """
    if not distribution:
        return 50.0

    percentile = (sum(1 for x in distribution if x <= value) / len(distribution)) * 100
    return percentile


def identify_critical_risks(features_df) -> List[Dict]:
    """
    Critical 위험 요인 식별

    Args:
        features_df: 특성 DataFrame

    Returns:
        [{
            'name': '현금소진일수',
            'value': 15.5,
            'threshold': 30.0,
            'explanation': '30일 이내에 현금이 고갈될 위험이 있습니다.'
        }, ...]
    """
    risks = []

    # 유동성 위기
    if features_df['유동비율'].iloc[0] < 1.0:
        risks.append({
            'name': '유동비율 부족',
            'value': features_df['유동비율'].iloc[0],
            'threshold': 1.0,
            'explanation': '단기 부채를 갚을 유동자산이 부족합니다. 즉시 유동성을 확보해야 합니다.'
        })

    if features_df['현금소진일수'].iloc[0] < 30:
        risks.append({
            'name': '현금 고갈 위험',
            'value': features_df['현금소진일수'].iloc[0],
            'threshold': 30.0,
            'explanation': f"현재 현금으로 {features_df['현금소진일수'].iloc[0]:.0f}일만 버틸 수 있습니다. 긴급하게 현금을 확보해야 합니다."
        })

    # 지급불능
    if features_df['이자보상배율'].iloc[0] < 1.0:
        risks.append({
            'name': '이자 지급 불능',
            'value': features_df['이자보상배율'].iloc[0],
            'threshold': 1.0,
            'explanation': '영업이익으로 이자비용을 감당할 수 없습니다. 차입금 상환 계획을 재검토해야 합니다.'
        })

    if features_df['자본잠식도'].iloc[0] > 0:
        risks.append({
            'name': '자본 잠식',
            'value': features_df['자본잠식도'].iloc[0],
            'threshold': 0.0,
            'explanation': '자본이 음수입니다. 즉시 자본 확충이 필요합니다.'
        })

    # 부채 과다
    if features_df['부채비율'].iloc[0] > 300:
        risks.append({
            'name': '과다 부채',
            'value': features_df['부채비율'].iloc[0],
            'threshold': 300.0,
            'explanation': '부채 비율이 300%를 초과했습니다. 부채 구조조정이 시급합니다.'
        })

    return risks


def identify_warnings(features_df) -> List[Dict]:
    """
    Warning 수준 위험 요인 식별

    Args:
        features_df: 특성 DataFrame

    Returns:
        경고 리스트
    """
    warnings = []

    # 유동성 경고
    if 1.0 <= features_df['유동비율'].iloc[0] < 1.5:
        warnings.append({
            'name': '유동비율 낮음',
            'value': features_df['유동비율'].iloc[0],
            'threshold': 1.5,
            'explanation': '유동비율이 150% 미만입니다. 유동자산을 늘리는 것이 좋습니다.'
        })

    if 30 <= features_df['현금소진일수'].iloc[0] < 90:
        warnings.append({
            'name': '현금 보유 부족',
            'value': features_df['현금소진일수'].iloc[0],
            'threshold': 90.0,
            'explanation': '현금 보유량이 3개월 미만입니다. 현금 확보를 권장합니다.'
        })

    # 수익성 경고
    if features_df['영업이익률'].iloc[0] < 0.05:
        warnings.append({
            'name': '낮은 수익성',
            'value': features_df['영업이익률'].iloc[0],
            'threshold': 0.05,
            'explanation': '영업이익률이 5% 미만입니다. 수익성 개선이 필요합니다.'
        })

    # 재무조작 의심
    if features_df['발생액비율'].iloc[0] > 0.1:
        warnings.append({
            'name': '높은 발생액 비율',
            'value': features_df['발생액비율'].iloc[0],
            'threshold': 0.1,
            'explanation': '이익이 현금으로 전환되지 않고 있습니다. 회계 정책을 검토하세요.'
        })

    return warnings


def generate_recommendations(features_df, financial_data: Dict) -> List[Dict]:
    """
    구체적 개선 권장사항 생성

    Args:
        features_df: 특성 DataFrame
        financial_data: 재무제표 데이터

    Returns:
        권장사항 리스트
    """
    recommendations = []

    # 1. 유동성 개선
    if features_df['유동비율'].iloc[0] < 1.5:
        현재유동비율 = features_df['유동비율'].iloc[0]
        필요유동자산증가 = (1.5 - 현재유동비율) * financial_data.get('유동부채', 0)

        recommendations.append({
            'title': '유동성 확보',
            'priority': 'High',
            'current_status': f"현재 유동비율: {현재유동비율:.2f}",
            'problem': '단기 부채 상환 능력이 부족합니다.',
            'solution': f"""
1. 유동자산을 {format_korean_number(필요유동자산증가)} 증가시키세요
   - 단기 금융상품 매각
   - 매출채권 조기 회수
   - 재고자산 정리

2. 또는 유동부채를 장기부채로 전환하세요
   - 단기차입금 → 장기차입금 전환
   - 만기 연장 협상
""",
            'expected_impact': '유동비율 150% 달성 시 부도 위험 20% 감소 예상'
        })

    # 2. 이자 부담 경감
    if features_df['이자보상배율'].iloc[0] < 2.0:
        recommendations.append({
            'title': '이자 부담 경감',
            'priority': 'High',
            'current_status': f"현재 이자보상배율: {features_df['이자보상배율'].iloc[0]:.2f}",
            'problem': '영업이익으로 이자를 감당하기 어렵습니다.',
            'solution': """
1. 고금리 차입금부터 상환하세요
2. 금리 재협상을 시도하세요
3. 영업이익을 늘리세요
   - 원가 절감
   - 판매가격 인상 검토
   - 신규 매출처 확보
""",
            'expected_impact': '이자보상배율 2.0 달성 시 부도 위험 15% 감소 예상'
        })

    # 3. 현금 흐름 개선
    if features_df['현금흐름적정성'].iloc[0] < 1.0:
        recommendations.append({
            'title': '현금 흐름 개선',
            'priority': 'Medium',
            'current_status': f"이익의 현금화율: {features_df['현금흐름적정성'].iloc[0]:.1%}",
            'problem': '이익이 현금으로 전환되지 않고 있습니다.',
            'solution': """
1. 매출채권 회수 기간 단축
   - 현재 회수 기간을 확인하세요
   - 조기 결제 할인 제공

2. 재고자산 회전율 향상
   - 재고 관리 최적화
   - 불용 재고 처분

3. 매입채무 지급 기간 연장 협상
""",
            'expected_impact': '현금 전환율 100% 달성 시 재무 건전성 10% 개선'
        })

    return recommendations[:3]  # 상위 3개만 반환
