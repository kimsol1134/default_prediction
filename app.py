"""
한국 기업 부도 예측 시스템 - Streamlit 앱

DART API 연동 및 실시간 부도 위험 분석
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
# 프로젝트 루트를 경로에 추가 (Streamlit Cloud 배포 시 deployment 폴더가 루트가 되므로 주석 처리)
# ROOT_DIR = Path(__file__).parent.parent
# sys.path.insert(0, str(ROOT_DIR))

from config import *
from src.dart_api import DartAPIClient, FinancialStatementParser
from src.domain_features import DomainFeatureGenerator
from src.models import BankruptcyPredictor

# Pickle 모델 로딩을 위해 transformer 클래스들 import (필수!)
# 모델이 __main__ 네임스페이스에서 저장되었을 경우 필요
try:
    from src.preprocessing.transformers import InfiniteHandler, LogTransformer, Winsorizer
except ImportError:
    pass  # 모듈이 없으면 무시 (휴리스틱 모드로 작동)

from src.visualization.charts import create_risk_gauge, create_shap_waterfall, create_shap_waterfall_real, create_radar_chart
from src.utils.helpers import (
    get_risk_level, format_korean_number,
    identify_critical_risks, identify_warnings, generate_recommendations
)
from src.utils.business_value import BusinessValueCalculator
import numpy as np

# 페이지 설정
st.set_page_config(**PAGE_CONFIG)

# 한글 폰트 설정
import matplotlib.pyplot as plt
plt.rc('font', family=KOREAN_FONT)
plt.rc('axes', unicode_minus=False)


# ========== 캐시된 리소스 ==========

@st.cache_resource
def load_predictor():
    """모델 로딩 (캐시) - Part3 파이프라인 우선 사용"""
    predictor = BankruptcyPredictor(
        pipeline_path=PIPELINE_PATH,  # Part3 전체 파이프라인
        model_path=MODEL_PATH,        # 레거시 모델 (fallback)
        scaler_path=SCALER_PATH,      # 레거시 스케일러 (fallback)
        use_pipeline=True             # 파이프라인 우선 사용
    )
    predictor.load_model()
    return predictor


@st.cache_data(ttl=3600)
def fetch_dart_data(company_name: str, year: str):
    """DART API 데이터 조회 (1시간 캐시)"""
    if not DART_API_KEY:
        st.error("❌ DART API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return None, None

    try:
        client = DartAPIClient(DART_API_KEY)

        # 기업 검색
        with st.spinner(f"'{company_name}' 검색 중..."):
            company = client.search_company(company_name)

        st.success(f"✓ {company['corp_name']} ({company['stock_code']}) 검색 완료")

        # 재무제표 조회
        with st.spinner(f"{year}년 재무제표 조회 중..."):
            statements = client.get_financial_statements(
                corp_code=company['corp_code'],
                bsns_year=year
            )

        st.success(f"✓ {year}년 재무제표 조회 완료")

        return company, statements

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        return None, None


# ========== 메인 앱 ==========

def main():
    """메인 앱"""

    # 헤더
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("---")

    # 사이드바 - 입력 방식 선택
    st.sidebar.header("📋 입력 방식 선택")

    input_method = st.sidebar.radio(
        "데이터 입력 방법",
        [
            "🔍 DART API 검색 (상장기업)",
            "📝 재무제표 직접 입력",
            "📂 샘플 데이터 사용"
        ]
    )

    # 변수 초기화
    company_info = None
    financial_data = None
    company_name = None
    year = None

    # ===== 입력 모드 1: DART API 검색 =====
    if input_method == "🔍 DART API 검색 (상장기업)":
        st.header("🔍 DART API 기업 검색")

        col1, col2 = st.columns([3, 1])

        with col1:
            company_name = st.text_input(
                "기업명 또는 종목코드",
                value="금양",
                help="예: 삼성전자, SK하이닉스, 005930"
            )

        with col2:
            # 동적으로 회계연도 생성 (전년도부터 과거 5년)
            from datetime import datetime
            current_year = datetime.now().year
            year_options = [str(current_year - 1 - i) for i in range(6)]  # 2024, 2023, 2022, 2021, 2020, 2019

            year = st.selectbox(
                "회계연도",
                options=year_options,
                index=0
            )

        if st.button("🚀 조회 및 분석 시작", type="primary"):
            # DART API 조회
            company, statements = fetch_dart_data(company_name, year)

            if company and statements:
                # 파싱
                parser = FinancialStatementParser()
                financial_data = parser.parse(statements)

                # 기업 정보 조회 (업종코드, 업력, 종업원수 등)
                client = DartAPIClient(DART_API_KEY)
                dart_company_info = client.get_company_info(company['corp_code'])

                # company_info 통합
                company_info = {
                    'corp_name': company['corp_name'],
                    'stock_code': company['stock_code'],
                    'year': year,
                    # DART company.json에서 가져온 정보
                    '업종코드': dart_company_info.get('업종코드', ''),
                    '업력': dart_company_info.get('업력', 10),
                    '종업원수': dart_company_info.get('종업원수', 100),
                    '외감여부': dart_company_info.get('외감여부', True),
                }

                # 분석 실행
                run_analysis(financial_data, company_info)

    # ===== 입력 모드 2: 직접 입력 =====
    elif input_method == "📝 재무제표 직접 입력":
        st.header("📝 재무제표 직접 입력")

        st.info("주요 재무 항목을 입력하세요 (단위: 백만원)")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("재무상태표")
            자산총계 = st.number_input("자산총계 (백만원)", value=1_000_000, step=10_000)
            부채총계 = st.number_input("부채총계 (백만원)", value=600_000, step=10_000)
            자본총계 = st.number_input("자본총계 (백만원)", value=400_000, step=10_000)
            유동자산 = st.number_input("유동자산 (백만원)", value=500_000, step=10_000)
            유동부채 = st.number_input("유동부채 (백만원)", value=300_000, step=10_000)
            현금 = st.number_input("현금및현금성자산 (백만원)", value=100_000, step=10_000)

        with col2:
            st.subheader("손익계산서")
            매출액 = st.number_input("매출액 (백만원)", value=2_000_000, step=10_000)
            매출원가 = st.number_input("매출원가 (백만원)", value=1_200_000, step=10_000)
            영업이익 = st.number_input("영업이익 (백만원)", value=200_000, step=10_000)
            당기순이익 = st.number_input("당기순이익 (백만원)", value=150_000, step=10_000)
            이자비용 = st.number_input("이자비용 (백만원)", value=20_000, step=1_000)
            영업활동현금흐름 = st.number_input("영업활동현금흐름 (백만원)", value=180_000, step=10_000)

        if st.button("🚀 분석 시작", type="primary"):
            financial_data = {
                '자산총계': 자산총계,
                '부채총계': 부채총계,
                '자본총계': 자본총계,
                '유동자산': 유동자산,
                '비유동자산': 자산총계 - 유동자산,
                '유동부채': 유동부채,
                '비유동부채': 부채총계 - 유동부채,
                '현금및현금성자산': 현금,
                '매출액': 매출액,
                '매출원가': 매출원가,
                '매출총이익': 매출액 - 매출원가,
                '영업이익': 영업이익,
                '당기순이익': 당기순이익,
                '이자비용': 이자비용,
                '영업활동현금흐름': 영업활동현금흐름,
                # 기타 기본값
                '단기금융상품': 0,
                '매출채권': 유동자산 * 0.2,
                '재고자산': 유동자산 * 0.1,
                '유형자산': (자산총계 - 유동자산) * 0.6,
                '무형자산': (자산총계 - 유동자산) * 0.1,
                '단기차입금': 유동부채 * 0.3,
                '장기차입금': (부채총계 - 유동부채) * 0.5,
                '판매비와관리비': 매출액 * 0.2,
                '매입채무': 유동부채 * 0.2,
            }

            company_info = {
                'corp_name': '직접입력 기업',
                'year': '2023'
            }

            run_analysis(financial_data, company_info)

    # ===== 입력 모드 3: 샘플 데이터 =====
    else:
        st.header("📂 샘플 데이터")

        st.info("샘플 기업 데이터로 시스템을 테스트해보세요.")

        sample_type = st.selectbox(
            "샘플 유형 선택",
            [
                "정상 기업 (부도 위험 낮음)",
                "주의 기업 (일부 위험 요소)",
                "위험 기업 (부도 위험 높음)"
            ]
        )

        if st.button("📊 샘플 분석", type="primary"):
            if "정상" in sample_type:
                financial_data = create_sample_data("normal")
                company_info = {'corp_name': '정상 샘플 기업', 'year': '2023'}
            elif "주의" in sample_type:
                financial_data = create_sample_data("caution")
                company_info = {'corp_name': '주의 샘플 기업', 'year': '2023'}
            else:
                financial_data = create_sample_data("risk")
                company_info = {'corp_name': '위험 샘플 기업', 'year': '2023'}

            run_analysis(financial_data, company_info)


def run_analysis(financial_data: dict, company_info: dict):
    """
    분석 실행 및 결과 표시

    Args:
        financial_data: 재무제표 데이터
        company_info: 기업 정보
    """
    st.markdown("---")
    st.header(f"📊 분석 결과: {company_info.get('corp_name', '기업')}")

    # 1. 특성 생성
    with st.spinner("도메인 특성 생성 중..."):
        generator = DomainFeatureGenerator()
        features_df = generator.generate_all_features(financial_data, company_info)

    st.success(f"✓ {len(features_df.columns)}개 특성 생성 완료")

    # 2. 예측
    with st.spinner("부도 위험 예측 중..."):
        predictor = load_predictor()
        result = predictor.predict(features_df)

    st.success("✓ 예측 완료")

    # ========== 섹션 1: 종합 평가 ==========
    display_overall_assessment(result, features_df, financial_data)

    # ========== 섹션 2: 위험 요인 분석 ==========
    display_risk_analysis(result, features_df)

    # ========== 섹션 3: 비즈니스 가치 분석 ==========
    display_business_value(result)

    # ========== 섹션 4: 개선 권장사항 ==========
    display_recommendations(features_df, financial_data)

    # ========== 섹션 5: 상세 특성 ==========
    display_detailed_features(features_df)

    # ========== 섹션 6: 재무제표 원본 ==========
    display_financial_statements(financial_data)


def display_overall_assessment(result: dict, features_df: pd.DataFrame, financial_data: dict):
    """섹션 1: 종합 평가"""
    st.markdown("## 🎯 종합 부도 위험 평가")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_prob = result['bankruptcy_probability']
        st.metric(
            label="부도 확률",
            value=f"{risk_prob*100:.1f}%",
            delta=f"{result['risk_level']} {result['risk_icon']}"
        )

    with col2:
        st.metric(
            label="위험 등급",
            value=result['risk_level'],
            delta=result['risk_icon']
        )

    with col3:
        건전성지수 = features_df.get('재무건전성지수', pd.Series([50])).iloc[0]
        st.metric(
            label="재무 건전성",
            value=f"{건전성지수:.0f}점",
            delta="100점 만점"
        )

    with col4:
        경보신호수 = int(features_df.get('조기경보신호수', pd.Series([0])).iloc[0])
        st.metric(
            label="조기경보신호",
            value=f"{경보신호수}개",
            delta="위험신호 개수"
        )

    # 게이지 차트
    st.plotly_chart(create_risk_gauge(risk_prob), use_container_width=True)

    # 메시지
    st.info(f"**분석 결과:** {result['risk_message']}")


def display_risk_analysis(result: dict, features_df: pd.DataFrame):
    """섹션 2: 위험 요인 분석"""
    st.markdown("---")
    st.markdown("## 🔍 위험 요인 상세 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Critical 리스크 (즉시 조치 필요)")

        critical_risks = identify_critical_risks(features_df)

        if critical_risks:
            for risk in critical_risks:
                st.error(
                    f"**{risk['name']}**: {risk['value']:.2f} "
                    f"(기준: {risk['threshold']:.2f})\n\n"
                    f"→ {risk['explanation']}"
                )
        else:
            st.success("✓ Critical 리스크 없음")

    with col2:
        st.markdown("### 🟡 Warning (개선 권장)")

        warnings = identify_warnings(features_df)

        if warnings:
            for warning in warnings:
                st.warning(
                    f"**{warning['name']}**: {warning['value']:.2f} "
                    f"(권장: {warning['threshold']:.2f})"
                )
        else:
            st.success("✓ Warning 없음")

    # SHAP Waterfall 차트
    st.markdown("### 📊 주요 위험 요인 기여도 (SHAP 분석)")
    if result.get('shap_values'):
        # 실제 SHAP 값 사용
        fig_shap = create_shap_waterfall_real(
            shap_values=np.array(result['shap_values']),
            feature_values=features_df.iloc[0],
            feature_names=result['feature_names'],
            base_value=result['shap_base_value']
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        # SHAP 값 없으면 간소화 버전 사용
        fig_shap = create_shap_waterfall(features_df.iloc[0])
        st.plotly_chart(fig_shap, use_container_width=True)
        st.info("ℹ️ 모델 로드 실패로 간소화된 분석을 표시합니다.")


def display_business_value(result: dict):
    """섹션 3: 비즈니스 가치 분석"""
    st.markdown("---")
    st.markdown("## 💰 비즈니스 가치 분석")

    calc = BusinessValueCalculator()
    value = calc.calculate_single_company(result['bankruptcy_probability'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("예상 손실", f"{value['expected_loss']:,.0f}원")

    with col2:
        st.metric("예상 수익", f"{value['expected_profit']:,.0f}원")

    with col3:
        delta_color = "normal" if value['net'] > 0 else "inverse"
        st.metric(
            "순 기대값",
            f"{value['net']:,.0f}원",
            delta="긍정적" if value['net'] > 0 else "부정적"
        )

    # 모델 성능 통계
    st.markdown("### 📊 모델 성능 (Test Set)")
    perf = calc.get_model_performance_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ROI", perf['roi'])

    with col2:
        st.metric("Payback", f"{perf['payback_months']}개월")

    with col3:
        st.metric("연간 절감", perf['annual_savings_krw'])

    with col4:
        st.metric("F2-Score", f"{perf['f2_score']:.2f}")

    st.info("""
    **💡 해석:**
    - **ROI 920%**: 모델 도입으로 투자 대비 9배 이상의 수익 창출
    - **Payback 1.3개월**: 모델 투자 비용을 1.3개월 내 회수
    - **연간 절감 460M KRW**: 잘못된 대출 결정 방지로 연간 4.6억원 절감
    """)


def display_recommendations(features_df: pd.DataFrame, financial_data: dict):
    """섹션 4: 개선 권장사항"""
    st.markdown("---")
    st.markdown("## 💡 실행 가능한 개선 권장사항")

    recommendations = generate_recommendations(features_df, financial_data)

    for i, rec in enumerate(recommendations, 1):
        with st.expander(
            f"권장사항 {i}: {rec['title']} (우선순위: {rec['priority']})",
            expanded=(i == 1)
        ):
            st.markdown(f"**현재 상태:**\n{rec['current_status']}")
            st.markdown(f"**문제점:**\n{rec['problem']}")
            st.markdown(f"**개선 방안:**{rec['solution']}")
            st.markdown(f"**예상 효과:**\n{rec['expected_impact']}")


def display_detailed_features(features_df: pd.DataFrame):
    """섹션 5: 상세 특성"""
    st.markdown("---")
    with st.expander("📋 생성된 특성 상세 보기"):
        st.markdown(f"총 {len(features_df.columns)}개 특성이 생성되었습니다.")

        # 카테고리별로 분류
        categories = {
            '유동성': [col for col in features_df.columns if any(kw in col for kw in ['유동', '현금', '운전자본'])],
            '지급불능': [col for col in features_df.columns if any(kw in col for kw in ['부채', '자본', '이자', '레버리지'])],
            '재무조작': [col for col in features_df.columns if any(kw in col for kw in ['발생액', '채권', '재고', '조작', '이익의질'])],
            '복합리스크': [col for col in features_df.columns if any(kw in col for kw in ['위험', '지수', '신호', '건전성'])]
        }

        for cat_name, cols in categories.items():
            if cols:
                st.markdown(f"**{cat_name} 특성 ({len(cols)}개)**")
                cat_df = features_df[cols].T
                cat_df.columns = ['값']
                st.dataframe(cat_df, use_container_width=True)


def display_financial_statements(financial_data: dict):
    """섹션 6: 재무제표 원본"""
    st.markdown("---")
    with st.expander("📋 재무제표 원본 데이터 보기"):
        # 재무상태표
        st.markdown("### 재무상태표")
        bs_data = {
            '항목': ['자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', '자본총계'],
            '금액 (백만원)': [
                financial_data.get('자산총계', 0),
                financial_data.get('유동자산', 0),
                financial_data.get('비유동자산', 0),
                financial_data.get('부채총계', 0),
                financial_data.get('유동부채', 0),
                financial_data.get('비유동부채', 0),
                financial_data.get('자본총계', 0)
            ]
        }
        st.dataframe(pd.DataFrame(bs_data), use_container_width=True)

        # 손익계산서
        st.markdown("### 손익계산서")
        is_data = {
            '항목': ['매출액', '매출원가', '매출총이익', '영업이익', '당기순이익'],
            '금액 (백만원)': [
                financial_data.get('매출액', 0),
                financial_data.get('매출원가', 0),
                financial_data.get('매출총이익', 0),
                financial_data.get('영업이익', 0),
                financial_data.get('당기순이익', 0)
            ]
        }
        st.dataframe(pd.DataFrame(is_data), use_container_width=True)


def create_sample_data(sample_type: str) -> dict:
    """샘플 데이터 생성"""
    if sample_type == "normal":
        return {
            '자산총계': 1_000_000, '부채총계': 400_000, '자본총계': 600_000,
            '유동자산': 600_000, '비유동자산': 400_000,
            '유동부채': 200_000, '비유동부채': 200_000,
            '현금및현금성자산': 200_000, '단기금융상품': 100_000,
            '매출채권': 150_000, '재고자산': 80_000,
            '유형자산': 250_000, '무형자산': 50_000,
            '단기차입금': 50_000, '장기차입금': 100_000,
            '매출액': 2_000_000, '매출원가': 1_200_000, '매출총이익': 800_000,
            '판매비와관리비': 400_000, '영업이익': 400_000,
            '이자비용': 10_000, '당기순이익': 300_000,
            '영업활동현금흐름': 350_000, '매입채무': 100_000,
        }
    elif sample_type == "caution":
        return {
            '자산총계': 1_000_000, '부채총계': 700_000, '자본총계': 300_000,
            '유동자산': 400_000, '비유동자산': 600_000,
            '유동부채': 400_000, '비유동부채': 300_000,
            '현금및현금성자산': 50_000, '단기금융상품': 20_000,
            '매출채권': 180_000, '재고자산': 100_000,
            '유형자산': 400_000, '무형자산': 100_000,
            '단기차입금': 150_000, '장기차입금': 250_000,
            '매출액': 1_500_000, '매출원가': 1_000_000, '매출총이익': 500_000,
            '판매비와관리비': 350_000, '영업이익': 150_000,
            '이자비용': 50_000, '당기순이익': 80_000,
            '영업활동현금흐름': 100_000, '매입채무': 120_000,
        }
    else:  # risk
        return {
            '자산총계': 1_000_000, '부채총계': 950_000, '자본총계': 50_000,
            '유동자산': 300_000, '비유동자산': 700_000,
            '유동부채': 500_000, '비유동부채': 450_000,
            '현금및현금성자산': 20_000, '단기금융상품': 5_000,
            '매출채권': 150_000, '재고자산': 80_000,
            '유형자산': 500_000, '무형자산': 100_000,
            '단기차입금': 250_000, '장기차입금': 400_000,
            '매출액': 1_000_000, '매출원가': 800_000, '매출총이익': 200_000,
            '판매비와관리비': 180_000, '영업이익': 20_000,
            '이자비용': 80_000, '당기순이익': -50_000,
            '영업활동현금흐름': 10_000, '매입채무': 150_000,
        }


# ========== 앱 실행 ==========

if __name__ == "__main__":
    main()
