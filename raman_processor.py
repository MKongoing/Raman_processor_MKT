import csv
import numpy as np
import pandas as pd
import os
import io
from scipy.integrate import trapezoid  # ← 修复：trapz 改为 trapezoid
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import streamlit as st

# 设置页面配置
st.set_page_config(page_title="拉曼光谱数据处理系统", layout="wide")


# ================= 核心处理函数 =================

def correct_baseline(wave, intensity, deg=1):
    """使用多项式拟合进行基线校正"""
    if len(wave) < 2:
        return intensity, np.zeros_like(intensity)
    p = np.polyfit(wave, intensity, deg=deg)
    baseline = np.polyval(p, wave)
    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)
    return corrected, baseline


def txt_to_csv_content(txt_content):
    """将 TXT 内容转换为 DataFrame（无需写文件）"""
    try:
        lines = [line for line in txt_content.decode('utf-8').splitlines() if line.strip()]

        if not lines:
            raise ValueError("输入文件为空")

        data = []
        for line in lines[1:]:  # 跳过标题行
            cleaned_line = line.replace('\t\t', '\t').strip()
            if cleaned_line:
                parts = cleaned_line.split('\t')
                if len(parts) >= 4:
                    data.append([float(p) for p in parts[:4]])

        df = pd.DataFrame(data, columns=["x", "y", "Wave", "Intensity"])
        return df.dropna()

    except Exception as e:
        st.error(f"文件解析失败：{e}")
        return None


def process_raman_data(df, d_range, g_range):
    """处理拉曼光谱数据"""
    try:
        xy_coords = df[['x', 'y']].drop_duplicates().values
        results = []
        total_points = len(xy_coords)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (x, y) in enumerate(xy_coords):
            spectrum = df[(df['x'] == x) & (df['y'] == y)]
            wave = spectrum['Wave'].values
            intensity = spectrum['Intensity'].values

            # 确保波长升序
            if len(wave) > 1 and wave[0] > wave[-1]:
                wave = wave[::-1]
                intensity = intensity[::-1]

            # 平滑处理
            if len(intensity) >= 11:
                intensity = savgol_filter(intensity, window_length=11, polyorder=2)

            intensity = np.maximum(intensity, 0)

            # D 峰积分
            d_mask = (wave >= d_range[0]) & (wave <= d_range[1])
            wave_d = wave[d_mask]
            intensity_d = intensity[d_mask]
            corrected_d, _ = correct_baseline(wave_d, intensity_d)
            d_area = trapezoid(corrected_d, wave_d) if len(wave_d) > 1 else 0.0

            # G 峰积分
            g_mask = (wave >= g_range[0]) & (wave <= g_range[1])
            wave_g = wave[g_mask]
            intensity_g = intensity[g_mask]
            corrected_g, _ = correct_baseline(wave_g, intensity_g)
            g_area = trapezoid(corrected_g, wave_g) if len(wave_g) > 1 else 0.0

            results.append([x, y, d_area, g_area])

            # 更新进度
            progress_bar.progress((i + 1) / total_points)
            status_text.text(f"正在处理：{i + 1}/{total_points}")

        status_text.text("✅ 处理完成！")
        return pd.DataFrame(results, columns=["x", "y", "D_Area", "G_Area"])

    except Exception as e:
        st.error(f"数据处理失败：{e}")
        return None


def plot_spectrum(df, x, y, d_range, g_range):
    """绘制指定坐标的光谱图"""
    try:
        spectrum = df[(df['x'] == x) & (df['y'] == y)]
        if spectrum.empty:
            return None

        wave = spectrum['Wave'].values
        intensity = spectrum['Intensity'].values

        if len(wave) > 1 and wave[0] > wave[-1]:
            wave = wave[::-1]
            intensity = intensity[::-1]

        if len(intensity) >= 11:
            intensity = savgol_filter(intensity, window_length=11, polyorder=2)

        intensity = np.maximum(intensity, 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wave, intensity, 'k-', linewidth=1, label='Spectrum')

        # D 峰
        d_mask = (wave >= d_range[0]) & (wave <= d_range[1])
        if np.any(d_mask):
            wave_d = wave[d_mask]
            intensity_d = intensity[d_mask]
            corrected_d, baseline_d = correct_baseline(wave_d, intensity_d)
            ax.fill_between(wave_d, corrected_d, color='red', alpha=0.3, label=f'D Peak')
            ax.plot(wave_d, baseline_d, 'r--', alpha=0.5)

        # G 峰
        g_mask = (wave >= g_range[0]) & (wave <= g_range[1])
        if np.any(g_mask):
            wave_g = wave[g_mask]
            intensity_g = intensity[g_mask]
            corrected_g, baseline_g = correct_baseline(wave_g, intensity_g)
            ax.fill_between(wave_g, corrected_g, color='green', alpha=0.3, label=f'G Peak')
            ax.plot(wave_g, baseline_g, 'g--', alpha=0.5)

        ax.set_title(f"Raman Spectrum at ({x}, {y})")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"绘图失败：{e}")
        return None


# ================= Streamlit 界面 =================

def main():
    # 初始化 Session State
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # 标题（只执行一次）
    st.title("🔬 拉曼光谱数据处理系统")
    st.markdown("---")

    # 侧边栏：参数设置
    with st.sidebar:
        st.header("⚙️ 参数设置")
        d_start = st.number_input("D 峰起始 (cm⁻¹)", value=1330)
        d_end = st.number_input("D 峰结束 (cm⁻¹)", value=1380)
        g_start = st.number_input("G 峰起始 (cm⁻¹)", value=1570)
        g_end = st.number_input("G 峰结束 (cm⁻¹)", value=1605)

        st.markdown("---")
        st.info("**使用说明：**\n1. 上传 TXT 文件\n2. 设置峰位范围\n3. 点击开始处理\n4. 查看结果和光谱图")

    # 主区域：文件上传
    uploaded_file = st.file_uploader("📁 上传 TXT 数据文件", type=['txt'],
                                     help="支持拖拽文件到此区域")

    if uploaded_file is not None:
        st.success(f"✅ 文件已加载：{uploaded_file.name}")

        # 显示数据预览（可折叠）
        with st.expander("📊 查看数据预览"):
            try:
                df_preview = txt_to_csv_content(uploaded_file.getvalue())
                if df_preview is not None:
                    st.dataframe(df_preview.head(10))
                    st.write(f"共 {len(df_preview)} 行数据")
            except Exception as e:
                st.error(f"预览失败：{e}")

        # 处理按钮
        if st.button("🚀 开始分析", type="primary", key="process_btn"):
            d_range = (d_start, d_end)
            g_range = (g_start, g_end)

            # 读取并转换数据
            df = txt_to_csv_content(uploaded_file.getvalue())

            if df is not None:
                st.session_state.df = df
                st.session_state.processed = False

                # 处理数据
                df_results = process_raman_data(df, d_range, g_range)

                if df_results is not None:
                    st.session_state.df_results = df_results
                    st.session_state.processed = True
                    st.session_state.d_range = d_range
                    st.session_state.g_range = g_range

                    # 显示统计结果
                    st.markdown("---")
                    st.subheader("📈 分析结果")

                    col1, col2 = st.columns(2)
                    with col1:
                        max_d = df_results.loc[df_results['D_Area'].idxmax()]
                        st.metric("最大 D 峰面积", f"{max_d['D_Area']:.2f}",
                                  f"坐标：({max_d['x']}, {max_d['y']})")
                    with col2:
                        max_g = df_results.loc[df_results['G_Area'].idxmax()]
                        st.metric("最大 G 峰面积", f"{max_g['G_Area']:.2f}",
                                  f"坐标：({max_g['x']}, {max_g['y']})")

                    # 下载结果
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下载结果 CSV",
                        data=csv,
                        file_name='raman_results.csv',
                        mime='text/csv',
                    )

                    # 显示结果表格
                    with st.expander("📋 查看完整结果表格"):
                        st.dataframe(df_results)

    # 光谱查看器（只有处理完成后显示）
    if st.session_state.processed and st.session_state.df is not None:
        st.markdown("---")
        st.subheader(" 光谱查看器")

        col1, col2 = st.columns([1, 3])
        with col1:
            # 坐标选择
            df_results = st.session_state.df_results
            x_val = st.number_input("X 坐标", value=float(df_results['x'].iloc[0]),
                                    key="x_input")
            y_val = st.number_input("Y 坐标", value=float(df_results['y'].iloc[0]),
                                    key="y_input")

            if st.button("绘制光谱", key="plot_btn"):
                fig = plot_spectrum(st.session_state.df, x_val, y_val,
                                    st.session_state.d_range, st.session_state.g_range)
                if fig:
                    st.session_state.current_fig = fig
                    st.session_state.show_plot = True

        with col2:
            if st.session_state.get('show_plot', False) and st.session_state.get('current_fig'):
                st.pyplot(st.session_state.current_fig)


if __name__ == "__main__":
    main()
