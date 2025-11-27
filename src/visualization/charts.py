"""
시각화 차트 생성 함수들

Plotly 기반 인터랙티브 차트
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_risk_gauge(risk_score: float) -> go.Figure:
    """
    위험도 게이지 차트

    Args:
        risk_score: 부도 확률 (0~1)

    Returns:
        Plotly Figure
    """
    # 위험 등급 결정
    if risk_score < 0.1:
        color = "#00CC00"  # 녹색
        level = "안전"
    elif risk_score < 0.3:
        color = "#FFCC00"  # 노란색
        level = "주의"
    elif risk_score < 0.6:
        color = "#FF9900"  # 주황색
        level = "경고"
    else:
        color = "#FF0000"  # 빨간색
        level = "위험"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"부도 위험도: {level}", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100], 'ticksuffix': "%"},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 10], 'color': "#E8F5E9"},
                {'range': [10, 30], 'color': "#FFF9C4"},
                {'range': [30, 60], 'color': "#FFE0B2"},
                {'range': [60, 100], 'color': "#FFCDD2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))

    fig.update_layout(
        height=400,
        font={'family': 'sans-serif'}
    )

    return fig


def create_shap_waterfall(
    feature_values: pd.Series,
    base_value: float = 0.015,
    top_n: int = 10
) -> go.Figure:
    """
    SHAP Waterfall 차트 (간소화 버전)

    주요 특성들이 부도 확률에 미치는 영향 시각화

    Args:
        feature_values: 특성값 Series
        base_value: 기준값 (평균 부도율)
        top_n: 상위 N개 특성만 표시

    Returns:
        Plotly Figure
    """
    # 중요 특성 선택 (임계값 기반)
    important_features = []

    # 유동성 위기
    if feature_values.get('유동비율', 1.0) < 1.0:
        important_features.append(('유동비율 부족', 0.05))
    if feature_values.get('현금소진일수', 90) < 30:
        important_features.append(('현금 고갈 위험', 0.08))

    # 지급불능
    if feature_values.get('이자보상배율', 2.0) < 1.0:
        important_features.append(('이자 지급 불능', 0.10))
    if feature_values.get('부채비율', 100) > 300:
        important_features.append(('과다 부채', 0.07))

    # 재무조작
    if feature_values.get('발생액비율', 0) > 0.1:
        important_features.append(('발생액 이상', 0.04))

    # 데이터 준비
    if not important_features:
        important_features = [('정상 범위', 0.0)]

    features, values = zip(*important_features)
    features = list(features)
    values = list(values)

    # 누적 계산
    cumulative = [base_value]
    for v in values:
        cumulative.append(cumulative[-1] + v)

    # Waterfall 차트
    fig = go.Figure(go.Waterfall(
        name="부도 확률 기여도",
        orientation="v",
        measure=["relative"] * len(features) + ["total"],
        x=features + ["최종 부도 확률"],
        y=values + [cumulative[-1] - base_value],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#00CC00"}},
        increasing={"marker": {"color": "#FF0000"}},
        totals={"marker": {"color": "#0066CC"}}
    ))

    fig.update_layout(
        title="주요 위험 요인 분석 (SHAP-style)",
        showlegend=False,
        height=500,
        font={'family': 'sans-serif'}
    )

    return fig


def create_shap_waterfall_real(
    shap_values: np.ndarray,
    feature_values: pd.Series,
    feature_names: List[str],
    base_value: float,
    max_display: int = 10
) -> go.Figure:
    """
    실제 SHAP 값 기반 Waterfall 차트

    Args:
        shap_values: SHAP 값 배열
        feature_values: 특성값 Series
        feature_names: 특성명 리스트
        base_value: 기준값 (expected_value)
        max_display: 표시할 최대 특성 개수

    Returns:
        Plotly Figure
    """
    # 절대값 기준 상위 N개 선택
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-max_display:][::-1]

    top_features = [feature_names[i] for i in top_indices]
    top_shap_values = [shap_values[i] for i in top_indices]

    # Waterfall 차트 생성
    fig = go.Figure(go.Waterfall(
        name="SHAP 기여도",
        orientation="v",
        measure=["absolute"] + ["relative"] * max_display + ["total"],
        x=["기준값"] + top_features + ["최종 예측"],
        y=[base_value] + top_shap_values + [sum(shap_values)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#51CF66"}},  # 초록 (위험 감소)
        increasing={"marker": {"color": "#FF6B6B"}},  # 빨강 (위험 증가)
        totals={"marker": {"color": "#4DABF7"}}       # 파랑 (최종값)
    ))

    fig.update_layout(
        title="SHAP 기여도 분석 (Waterfall)",
        showlegend=False,
        height=500,
        font={'family': 'sans-serif'},
        yaxis_title="부도 확률 기여도"
    )

    return fig


def create_radar_chart(
    company_features: pd.DataFrame,
    industry_avg: Optional[Dict] = None
) -> go.Figure:
    """
    5대 재무 지표 레이더 차트 (업계 평균 비교)

    Args:
        company_features: 기업 특성 DataFrame
        industry_avg: 업계 평균값 딕셔너리

    Returns:
        Plotly Figure
    """
    # 5대 지표 선택
    categories = ['유동성', '안정성', '수익성', '활동성', '성장성']

    # 기업 값 계산
    company_values = [
        min(100, max(0, company_features.get('유동비율', [1.0]).iloc[0] * 100)),  # 유동성
        min(100, max(0, 100 - company_features.get('부채비율', [100]).iloc[0] / 3)),  # 안정성
        min(100, max(0, company_features.get('영업이익률', [0]).iloc[0] * 100 * 5)),  # 수익성
        min(100, max(0, 50)),  # 활동성 (간소화)
        min(100, max(0, 50))   # 성장성 (간소화)
    ]

    # 업계 평균 (기본값)
    if industry_avg is None:
        industry_values = [70, 60, 50, 50, 50]
    else:
        industry_values = [
            industry_avg.get('유동성', 70),
            industry_avg.get('안정성', 60),
            industry_avg.get('수익성', 50),
            industry_avg.get('활동성', 50),
            industry_avg.get('성장성', 50)
        ]

    fig = go.Figure()

    # 기업 데이터
    fig.add_trace(go.Scatterpolar(
        r=company_values,
        theta=categories,
        fill='toself',
        name='해당 기업',
        line_color='#0066CC'
    ))

    # 업계 평균
    fig.add_trace(go.Scatterpolar(
        r=industry_values,
        theta=categories,
        fill='toself',
        name='업계 평균',
        line_color='#999999',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="5대 재무 지표 비교",
        height=500,
        font={'family': 'sans-serif'}
    )

    return fig


def create_timeline_chart(years: List[str], metrics: Dict[str, List[float]]) -> go.Figure:
    """
    다년도 추이 차트

    Args:
        years: 연도 리스트 ['2021', '2022', '2023']
        metrics: {'매출액': [100, 120, 150], '영업이익': [10, 12, 18]}

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for metric_name, values in metrics.items():
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=metric_name,
            line=dict(width=3),
            marker=dict(size=10)
        ))

    fig.update_layout(
        title="주요 지표 추이",
        xaxis_title="연도",
        yaxis_title="값",
        hovermode='x unified',
        height=400,
        font={'family': 'sans-serif'}
    )

    return fig
