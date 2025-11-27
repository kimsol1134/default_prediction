"""
DART API 응답을 표준 재무제표 포맷으로 변환

입력: DART API JSON 응답
출력: 표준화된 재무제표 딕셔너리

주요 기능:
1. 계정과목 매핑 (DART 표준 → 프로젝트 표준)
2. 금액 단위 변환 (원 → 백만원)
3. 결측 항목 처리
4. 데이터 검증 (음수 자산 등 이상치 탐지)
"""

import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


class FinancialStatementParser:
    """재무제표 파싱 및 표준화"""

    # 계정과목 매핑 테이블 (DART 표준 → 프로젝트 표준)
    ACCOUNT_MAPPING = {
        # === 재무상태표 (Balance Sheet) ===
        # 자산
        '유동자산': ['유동자산', '당좌자산'],
        '현금및현금성자산': ['현금및현금성자산', '현금및현금성자산(유동)', '현금및예금'],
        '단기금융상품': ['단기금융상품', '단기투자자산'],
        '매출채권': ['매출채권', '매출채권 및 기타채권', '매출채권및기타채권', '단기매출채권'],
        '재고자산': ['재고자산'],
        '비유동자산': ['비유동자산', '고정자산'],
        '유형자산': ['유형자산', '유형자산(순액)'],
        '무형자산': ['무형자산'],
        '투자자산': ['투자자산', '장기투자증권'],
        '자산총계': ['자산총계', '자산총액'],

        # 부채
        '유동부채': ['유동부채'],
        '단기차입금': ['단기차입금', '단기借入金'],
        '매입채무': ['매입채무', '매입채무 및 기타채무'],
        '비유동부채': ['비유동부채', '고정부채'],
        '장기차입금': ['장기차입금', '장기借入金'],
        '사채': ['사채'],
        '부채총계': ['부채총계', '부채총액'],

        # 자본
        '자본금': ['자본금'],
        '이익잉여금': ['이익잉여금'],
        '자본잉여금': ['자본잉여금'],
        '자본총계': ['자본총계', '자본총액'],

        # === 손익계산서 (Income Statement) ===
        '매출액': ['매출액', '수익(매출액)', '영업수익'],
        '매출원가': ['매출원가'],
        '매출총이익': ['매출총이익'],
        '판매비와관리비': ['판매비와관리비', '판매비와 관리비', '판매비 및 일반관리비'],
        '영업이익': ['영업이익(손실)', '영업이익', '영업손익'],
        '영업외수익': ['영업외수익', '기타수익'],
        '영업외비용': ['영업외비용', '기타비용'],
        '이자수익': ['이자수익'],
        '이자비용': ['이자비용', '금융원가', '이자비용(금융원가)'],
        '법인세비용차감전순이익': ['법인세비용차감전순이익(손실)', '법인세비용차감전순이익'],
        '법인세비용': ['법인세비용'],
        '당기순이익': ['당기순이익(손실)', '당기순이익', '당기순손익'],

        # === 현금흐름표 (Cash Flow Statement) ===
        '영업활동현금흐름': [
            '영업활동으로인한현금흐름',
            '영업활동현금흐름',
            '영업활동으로 인한 현금흐름',
            '영업활동 현금흐름'
        ],
        '투자활동현금흐름': [
            '투자활동으로인한현금흐름',
            '투자활동현금흐름',
            '투자활동으로 인한 현금흐름'
        ],
        '재무활동현금흐름': [
            '재무활동으로인한현금흐름',
            '재무활동현금흐름',
            '재무활동으로 인한 현금흐름'
        ],
    }

    # 필수 계정과목 (검증용)
    REQUIRED_ACCOUNTS = {
        'balance_sheet': ['자산총계', '부채총계', '자본총계', '유동자산', '유동부채'],
        'income_statement': ['매출액', '영업이익', '당기순이익'],
        'cash_flow': ['영업활동현금흐름']
    }

    def __init__(self, unit_conversion: float = 1_000_000):
        """
        Args:
            unit_conversion: 금액 단위 변환 (기본: 백만원)
                - 1: 원 단위 (변환 없음)
                - 1_000: 천원 단위
                - 1_000_000: 백만원 단위 (기본)
        """
        self.unit_conversion = unit_conversion

    def parse(self, dart_response: Dict) -> Dict:
        """
        DART API 응답 파싱

        Args:
            dart_response: get_financial_statements() 응답
                {
                    'balance_sheet': {...},
                    'income_statement': {...},
                    'cash_flow': {...},
                    'metadata': {...}
                }

        Returns:
            표준화된 재무제표
            {
                '유동자산': 1000000,  # 백만원 단위
                '유동부채': 500000,
                '매출액': 3000000,
                ...
                '_metadata': {...}
            }
        """
        standardized = {}

        # 각 재무제표 통합 및 매핑
        for statement_type in ['balance_sheet', 'income_statement', 'cash_flow']:
            raw_data = dart_response.get(statement_type, {})
            mapped_data = self._map_accounts(raw_data)
            # 0이 아닌 값만 업데이트 (나중 재무제표가 이전 값을 0으로 덮어쓰는 것 방지)
            for key, value in mapped_data.items():
                if value != 0 or key not in standardized:
                    standardized[key] = value

        # 금액 단위 변환
        if self.unit_conversion != 1:
            for key in standardized:
                if isinstance(standardized[key], (int, float)):
                    standardized[key] = standardized[key] / self.unit_conversion

        # 메타데이터 추가
        standardized['_metadata'] = dart_response.get('metadata', {})

        # 파생 항목 계산
        standardized = self._calculate_derived_accounts(standardized)

        logger.info(f"재무제표 파싱 완료: {len(standardized)}개 항목")

        return standardized

    def _map_accounts(self, raw_data: Dict) -> Dict:
        """
        계정과목 매핑 (DART → 프로젝트 표준)

        Args:
            raw_data: DART API 원본 데이터 {'계정명': 금액}

        Returns:
            매핑된 데이터 {'표준계정명': 금액}
        """
        mapped = {}

        for standard_name, dart_names in self.ACCOUNT_MAPPING.items():
            # 여러 가능한 이름 중 첫 번째로 발견되는 값 사용
            for dart_name in dart_names:
                if dart_name in raw_data:
                    mapped[standard_name] = raw_data[dart_name]
                    break
            else:
                # 매핑되는 항목이 없으면 0으로 설정
                mapped[standard_name] = 0.0

        return mapped

    def _calculate_derived_accounts(self, data: Dict) -> Dict:
        """
        파생 계정과목 계산

        예:
        - 당좌자산 = 유동자산 - 재고자산
        - 순차입금 = (단기차입금 + 장기차입금) - 현금및현금성자산
        """
        # 당좌자산
        if '당좌자산' not in data or data.get('당좌자산', 0) == 0:
            data['당좌자산'] = data.get('유동자산', 0) - data.get('재고자산', 0)

        # 순차입금
        data['순차입금'] = (
            data.get('단기차입금', 0) +
            data.get('장기차입금', 0) -
            data.get('현금및현금성자산', 0)
        )

        # 차입금총계
        data['차입금총계'] = data.get('단기차입금', 0) + data.get('장기차입금', 0)

        # 매출총이익 (없는 경우 계산)
        if data.get('매출총이익', 0) == 0:
            data['매출총이익'] = data.get('매출액', 0) - data.get('매출원가', 0)

        return data

    def validate(self, financial_data: Dict) -> Tuple[bool, List[str]]:
        """
        재무제표 검증

        검증 항목:
        1. 필수 계정과목 존재 여부
        2. 재무상태표 항등식: 자산 = 부채 + 자본
        3. 음수 자산/부채/자본 체크
        4. 이상치 탐지

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # 1. 필수 계정과목 체크
        required = [
            '자산총계', '부채총계', '자본총계',
            '유동자산', '유동부채',
            '매출액', '당기순이익'
        ]

        for account in required:
            if account not in financial_data or financial_data[account] == 0:
                errors.append(f"필수 계정과목 누락: {account}")

        # 2. 재무상태표 항등식 검증
        assets = financial_data.get('자산총계', 0)
        liabilities = financial_data.get('부채총계', 0)
        equity = financial_data.get('자본총계', 0)

        balance_diff = abs(assets - (liabilities + equity))
        tolerance = assets * 0.01  # 1% 허용 오차

        if balance_diff > tolerance:
            errors.append(
                f"재무상태표 불균형: 자산({assets:.0f}) ≠ 부채({liabilities:.0f}) + 자본({equity:.0f}), "
                f"차이: {balance_diff:.0f}"
            )

        # 3. 음수 자산/부채 체크 (자본은 음수 가능 - 자본잠식)
        if assets < 0:
            errors.append(f"음수 자산: {assets}")
        if liabilities < 0:
            errors.append(f"음수 부채: {liabilities}")

        # 4. 이상치 탐지
        if assets > 0 and liabilities / assets > 10:
            errors.append(f"부채비율 이상: {liabilities / assets * 100:.1f}% (>1000%)")

        # 매출액 음수 체크
        revenue = financial_data.get('매출액', 0)
        if revenue < 0:
            errors.append(f"음수 매출액: {revenue}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("재무제표 검증 성공")
        else:
            logger.warning(f"재무제표 검증 실패: {len(errors)}개 오류")
            for error in errors:
                logger.warning(f"  - {error}")

        return is_valid, errors

    def get_summary(self, financial_data: Dict) -> Dict:
        """
        재무제표 요약 정보 생성

        Returns:
            {
                '기업규모': '대기업',
                '매출액': 1000000,
                '영업이익률': 0.15,
                '부채비율': 150.0,
                '유동비율': 120.0
            }
        """
        summary = {}

        # 기본 정보
        revenue = financial_data.get('매출액', 0)
        assets = financial_data.get('자산총계', 0)

        summary['매출액'] = revenue
        summary['자산총계'] = assets

        # 기업 규모 분류 (매출액 기준)
        if revenue >= 1_000_000:  # 1조원 이상 (백만원 단위)
            summary['기업규모'] = '대기업'
        elif revenue >= 100_000:  # 1000억원 이상
            summary['기업규모'] = '중견기업'
        else:
            summary['기업규모'] = '중소기업'

        # 주요 재무비율
        operating_income = financial_data.get('영업이익', 0)
        if revenue > 0:
            summary['영업이익률'] = (operating_income / revenue) * 100

        net_income = financial_data.get('당기순이익', 0)
        if revenue > 0:
            summary['순이익률'] = (net_income / revenue) * 100

        liabilities = financial_data.get('부채총계', 0)
        equity = financial_data.get('자본총계', 0)
        if equity > 0:
            summary['부채비율'] = (liabilities / equity) * 100

        current_assets = financial_data.get('유동자산', 0)
        current_liabilities = financial_data.get('유동부채', 0)
        if current_liabilities > 0:
            summary['유동비율'] = (current_assets / current_liabilities) * 100

        return summary
