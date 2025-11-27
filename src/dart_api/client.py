"""
DART Open API 클라이언트

주요 기능:
1. 기업 검색 (회사명 → 종목코드 변환)
2. 재무제표 조회 (단일 회계연도)
3. 다년도 재무제표 조회 (추이 분석용)
4. API Rate Limit 관리
5. 에러 핸들링 및 재시도 로직

필수 API 엔드포인트:
- 공시검색: /api/company.json (기업 기본정보)
- 재무제표: /api/fnlttSinglAcntAll.json (단일회사 전체 재무제표)
"""

import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import zipfile
import io
import xml.etree.ElementTree as ET

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DartAPIClient:
    """DART API 클라이언트"""

    BASE_URL = "https://opendart.fss.or.kr/api"
    RATE_LIMIT_DELAY = 1.0  # 초당 1회 제한

    def __init__(self, api_key: str):
        """
        Args:
            api_key: DART API 키 (환경 변수에서 로드)
        """
        if not api_key:
            raise ValueError("DART API 키가 제공되지 않았습니다. .env 파일에 DART_API_KEY를 설정하세요.")

        self.api_key = api_key
        self.last_request_time = 0
        self.session = requests.Session()
        self.corp_codes = {}  # 기업 코드 캐시 (회사명 -> corp_code 매핑)

        # corpCode.xml 다운로드 및 파싱
        self._load_corp_codes()

    def _load_corp_codes(self):
        """
        DART에서 제공하는 corpCode.xml 파일을 다운로드하고 파싱
        회사명 -> corp_code 매핑 테이블 생성
        """
        try:
            logger.info("기업 코드 목록(corpCode.xml) 다운로드 중...")

            # corpCode.zip 다운로드
            url = f"{self.BASE_URL}/corpCode.xml"
            params = {'crtfc_key': self.api_key}

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            # ZIP 파일 압축 해제
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # CORPCODE.xml 파일 읽기 (대문자 주의)
                xml_content = zip_file.read('CORPCODE.xml')

            # XML 파싱
            root = ET.fromstring(xml_content)

            # 기업 코드 매핑 생성
            for company in root.findall('list'):
                corp_code = company.find('corp_code').text
                corp_name = company.find('corp_name').text
                stock_code = company.find('stock_code')

                if corp_name and corp_code:
                    self.corp_codes[corp_name] = {
                        'corp_code': corp_code,
                        'corp_name': corp_name,
                        'stock_code': stock_code.text if stock_code is not None and stock_code.text else None
                    }

            logger.info(f"✓ 기업 코드 {len(self.corp_codes):,}개 로드 완료")

        except Exception as e:
            logger.error(f"기업 코드 로드 실패: {str(e)}")
            logger.warning("기업 검색 기능이 제한될 수 있습니다.")

    def _handle_rate_limit(self):
        """API Rate Limit 관리 (1초당 1회 제한)"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - time_since_last_request
            logger.debug(f"Rate limit 대기: {sleep_time:.2f}초")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Dict,
        max_retries: int = 3
    ) -> Dict:
        """
        API 요청 실행 (에러 핸들링 및 재시도 포함)

        Args:
            endpoint: API 엔드포인트
            params: 요청 파라미터
            max_retries: 최대 재시도 횟수

        Returns:
            API 응답 JSON

        Raises:
            requests.exceptions.Timeout: 타임아웃
            requests.exceptions.HTTPError: HTTP 에러
            ValueError: 잘못된 응답
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params['crtfc_key'] = self.api_key

        for attempt in range(max_retries):
            try:
                self._handle_rate_limit()

                logger.info(f"API 요청: {endpoint}, 시도: {attempt + 1}/{max_retries}")
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                # DART API 에러 체크
                if data.get('status') == '000':
                    return data
                elif data.get('status') == '013':
                    raise ValueError(f"검색 결과가 없습니다: {params}")
                elif data.get('status') == '020':
                    raise ValueError("API 키가 유효하지 않습니다.")
                else:
                    error_msg = data.get('message', '알 수 없는 에러')
                    raise ValueError(f"DART API 에러 ({data.get('status')}): {error_msg}")

            except requests.exceptions.Timeout:
                logger.warning(f"타임아웃 발생 (시도 {attempt + 1})")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 지수 백오프

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning("Rate limit 초과, 재시도 중...")
                    time.sleep(5)
                else:
                    raise

        raise Exception("최대 재시도 횟수 초과")

    def search_company(self, company_name: str) -> Dict:
        """
        기업명으로 종목코드 검색

        Args:
            company_name: 기업명 (예: "삼성전자") 또는 종목코드 (예: "005930")

        Returns:
            {
                'corp_code': '00126380',
                'corp_name': '삼성전자',
                'stock_code': '005930'
            }

        Raises:
            ValueError: 기업을 찾을 수 없음
        """
        try:
            # 1. 정확한 회사명으로 검색
            if company_name in self.corp_codes:
                result = self.corp_codes[company_name]
                logger.info(f"기업 검색 성공: {result['corp_name']} ({result.get('stock_code', 'N/A')})")
                return result

            # 2. 종목코드로 검색
            if company_name.isdigit():
                for corp_info in self.corp_codes.values():
                    if corp_info.get('stock_code') == company_name:
                        logger.info(f"종목코드 검색 성공: {corp_info['corp_name']} ({company_name})")
                        return corp_info

            # 3. 부분 일치 검색 (회사명에 포함)
            matches = []
            for corp_name, corp_info in self.corp_codes.items():
                if company_name in corp_name or corp_name in company_name:
                    matches.append(corp_info)

            if matches:
                # 가장 짧은 이름 (가장 유사한 결과) 반환
                result = min(matches, key=lambda x: len(x['corp_name']))
                logger.info(f"부분 일치 검색 성공: {result['corp_name']} ({result.get('stock_code', 'N/A')})")
                return result

            # 검색 실패
            raise ValueError(
                f"'{company_name}' 기업을 찾을 수 없습니다.\n"
                f"정확한 회사명이나 종목코드를 입력하세요.\n"
                f"예: '삼성전자', 'SK하이닉스', '005930'"
            )

        except Exception as e:
            logger.error(f"기업 검색 실패: {str(e)}")
            raise

    def get_financial_statements(
        self,
        corp_code: str,
        bsns_year: str,
        reprt_code: str = "11011"  # 사업보고서
    ) -> Dict:
        """
        재무제표 조회

        Args:
            corp_code: 고유번호 (8자리)
            bsns_year: 사업연도 (YYYY)
            reprt_code: 보고서 코드
                - 11011: 사업보고서
                - 11012: 반기보고서
                - 11013: 1분기보고서
                - 11014: 3분기보고서

        Returns:
            {
                'balance_sheet': {...},      # 재무상태표
                'income_statement': {...},   # 손익계산서
                'cash_flow': {...}           # 현금흐름표
            }

        Raises:
            ValueError: 재무제표를 찾을 수 없음
        """
        endpoint = "fnlttSinglAcntAll.json"
        params = {
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code,
            'fs_div': 'CFS'  # CFS: 연결재무제표, OFS: 개별재무제표
        }

        try:
            data = self._make_request(endpoint, params)

            if not data.get('list'):
                # 연결재무제표가 없으면 개별재무제표 시도
                logger.warning("연결재무제표 없음, 개별재무제표 조회 중...")
                params['fs_div'] = 'OFS'
                data = self._make_request(endpoint, params)

                if not data.get('list'):
                    raise ValueError(f"재무제표를 찾을 수 없습니다: {corp_code}, {bsns_year}")

            # 계정과목별로 분류
            statements = {
                'balance_sheet': {},
                'income_statement': {},
                'cash_flow': {},
                'metadata': {
                    'corp_code': corp_code,
                    'bsns_year': bsns_year,
                    'reprt_code': reprt_code,
                    'fs_div': params['fs_div']
                }
            }

            for item in data['list']:
                account_nm = item.get('account_nm')  # 계정명
                thstrm_amount = item.get('thstrm_amount')  # 당기금액
                sj_div = item.get('sj_div')  # 재무제표구분 (BS: 재무상태표, IS: 손익계산서, CF: 현금흐름표)

                if not account_nm or not thstrm_amount:
                    continue

                # 숫자로 변환 (쉼표 제거)
                try:
                    amount = float(thstrm_amount.replace(',', ''))
                except (ValueError, AttributeError):
                    amount = 0

                # 재무제표 구분에 따라 저장
                if sj_div == 'BS':
                    statements['balance_sheet'][account_nm] = amount
                elif sj_div == 'IS':
                    statements['income_statement'][account_nm] = amount
                elif sj_div == 'CF':
                    statements['cash_flow'][account_nm] = amount

            logger.info(f"재무제표 조회 성공: {bsns_year}년")
            logger.info(f"  - 재무상태표: {len(statements['balance_sheet'])}개 항목")
            logger.info(f"  - 손익계산서: {len(statements['income_statement'])}개 항목")
            logger.info(f"  - 현금흐름표: {len(statements['cash_flow'])}개 항목")

            return statements

        except Exception as e:
            logger.error(f"재무제표 조회 실패: {str(e)}")
            raise

    def get_multi_year_statements(
        self,
        corp_code: str,
        years: List[str]
    ) -> Dict[str, Dict]:
        """
        다년도 재무제표 조회 (추이 분석용)

        Args:
            corp_code: 고유번호
            years: 사업연도 리스트 (예: ['2021', '2022', '2023'])

        Returns:
            {
                '2021': {...},
                '2022': {...},
                '2023': {...}
            }
        """
        results = {}

        for year in years:
            try:
                statements = self.get_financial_statements(corp_code, year)
                results[year] = statements
            except Exception as e:
                logger.warning(f"{year}년 재무제표 조회 실패: {str(e)}")
                results[year] = None

        return results

    def get_company_info(self, corp_code: str) -> Dict:
        """
        기업 개황 정보 조회 (company.json API)

        업종코드, 설립일, 종업원수 등 기본 정보 조회

        Args:
            corp_code: 고유번호 (8자리)

        Returns:
            {
                '업종': '반도체',
                '업종코드': 'C26',  # KSIC 코드
                '설립일': '1969-01-13',
                '업력': 55,  # 설립 후 경과 년수
                '종업원수': 120000,
                '외감여부': True,  # 상장사는 True
                ...
            }

        Note:
            - DART API의 company.json 엔드포인트 사용
            - 상장사만 정확한 정보 제공
            - 비상장사는 일부 정보 누락 가능
        """
        endpoint = "company.json"
        params = {'corp_code': corp_code}

        try:
            data = self._make_request(endpoint, params)

            # 기본 정보 추출
            company_info = {}

            # 업종코드 (DART API 필드명: induty_code)
            # KSIC 표준산업분류 코드 (예: C26 = 전자부품,컴퓨터,영상,음향및통신장비 제조업)
            if 'induty_code' in data and data['induty_code']:
                company_info['업종코드'] = data['induty_code']
                logger.info(f"  업종코드: {data['induty_code']}")
            else:
                company_info['업종코드'] = ''
                logger.warning("  업종코드 없음")

            # 설립일 및 업력 계산 (DART API 필드명: est_dt, 형식: YYYYMMDD)
            if 'est_dt' in data and data['est_dt']:
                est_date = data['est_dt']
                try:
                    est_year = int(est_date[:4]) if len(est_date) >= 4 else 1990
                    current_year = datetime.now().year
                    company_info['설립일'] = est_date
                    company_info['업력'] = current_year - est_year
                    logger.info(f"  설립일: {est_date} → 업력 {company_info['업력']}년")
                except (ValueError, TypeError):
                    company_info['설립일'] = None
                    company_info['업력'] = 10  # 기본값
                    logger.warning("  설립일 파싱 실패, 기본값 10년 사용")
            else:
                company_info['설립일'] = None
                company_info['업력'] = 10  # 기본값
                logger.warning("  설립일 없음, 기본값 10년 사용")

            # 종업원수 (DART company.json에는 없음, 별도 API 필요)
            # 기본값 사용 (대기업은 일반적으로 10,000명 이상)
            company_info['종업원수'] = 10000  # 상장사 기본값 (조정 가능)

            # 외감여부 (상장사는 True)
            # stock_code가 있으면 상장사로 간주
            stock_code = self._get_stock_code_from_corp_code(corp_code)
            company_info['외감여부'] = bool(stock_code)

            # 추가 정보
            company_info['corp_code'] = corp_code
            company_info['corp_name'] = data.get('corp_name', '')
            company_info['ceo_nm'] = data.get('ceo_nm', '')
            company_info['jurir_no'] = data.get('jurir_no', '')  # 법인등록번호
            company_info['stock_code'] = stock_code

            logger.info(f"✓ 기업 정보 조회 성공: {company_info.get('corp_name', '')}")
            logger.info(f"  업종코드: {company_info.get('업종코드') or 'N/A'}")
            logger.info(f"  업력: {company_info.get('업력')}년")
            logger.info(f"  종업원수: {company_info.get('종업원수'):,}명 (기본값)")
            logger.info(f"  외감여부: {company_info.get('외감여부')}")

            return company_info

        except Exception as e:
            logger.warning(f"기업 정보 조회 실패: {str(e)}")
            logger.warning("기본값으로 대체합니다.")

            # 실패 시 기본값 반환
            return {
                'corp_code': corp_code,
                '업종코드': '',
                '업력': 10,
                '종업원수': 10000,  # 상장사 기본값
                '외감여부': True,
            }

    def _get_stock_code_from_corp_code(self, corp_code: str) -> Optional[str]:
        """corp_code로부터 stock_code 조회 (캐시에서)"""
        for corp_info in self.corp_codes.values():
            if corp_info.get('corp_code') == corp_code:
                return corp_info.get('stock_code')
        return None
