import csv
import numpy as np
import pandas as pd
import os
import io
from scipy.integrate import trapezoid
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
    """将 TXT 内容转换为 DataFrame"""
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
            
            if len(wave) > 1 and wave[0] > wave[-1]:
                wave = wave[::-1]
                intensity = intensity[::-1]
            
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
            area_d = trapezoid(corrected_d, wave_d)
            ax.fill_between(wave_d, corrected_d, color='red', alpha=0.3, 
                           label=f'D Peak (Area: {area_d:.2f})')
            ax.plot(wave_d, baseline_d, 'r--', alpha=0.5)
        
        # G 峰
        g_mask = (wave >= g_range[0]) & (wave <= g_range[1])
        if np.any(g_mask):
            wave_g = wave[g_mask]
            intensity_g = intensity[g_mask]
            corrected_g, baseline_g = correct_baseline(wave_g, intensity_g)
            area_g = trapezoid(corrected_g, wave_g)
            ax.fill_between(wave_g, corrected_g, color='green', alpha=0.3,
                           label=f'G Peak (Area: {area_g:.2f})')
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

def select_max_d():
    """回调函数：选择最大 D 峰坐标"""
    if st.session_state.df_results is not None:
        max_d = st.session_state.df_results.loc[
            st.session_state.df_results['D_Area'].idxmax()]
        st.session_state.selected_x = max_d['x']
        st.session_state.selected_y = max_d['y']


def select_max_g():
    """回调函数：选择最大 G 峰坐标"""
    if st.session_state.df_results is not None:
        max_g = st.session_state.df_results.loc[
            st.session_state.df_results['G_Area'].idxmax()]
        st.session_state.selected_x = max_g['x']
        st.session_state.selected_y = max_g['y']


def main():
    # ========== 初始化 Session State ==========
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_results' not in st.session_state:
        st.session_state.df_results = None
    if 'd_range' not in st.session_state:
        st.session_state.d_range = (1330, 1380)
    if 'g_range' not in st.session_state:
        st.session_state.g_range = (1570, 1605)
    if 'current_fig' not in st.session_state:
        st.session_state.current_fig = None
    if 'selected_x' not in st.session_state:
        st.session_state.selected_x = None
    if 'selected_y' not in st.session_state:
        st.session_state.selected_y = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # ========== 标题 ==========
    st.title("🔬 拉曼光谱数据处理系统")
    st.markdown("---")
    
    # ========== 侧边栏：参数设置 ==========
    with st.sidebar:
        st.header("⚙️ 参数设置")
        d_start = st.number_input("D 峰起始 (cm⁻¹)", value=1330, key="d_start_sidebar")
        d_end = st.number_input("D 峰结束 (cm⁻¹)", value=1380, key="d_end_sidebar")
        g_start = st.number_input("G 峰起始 (cm⁻¹)", value=1570, key="g_start_sidebar")
        g_end = st.number_input("G 峰结束 (cm⁻¹)", value=1605, key="g_end_sidebar")
        
        st.markdown("---")
        st.info("**使用说明：**\n1. 上传 TXT 文件\n2. 设置峰位范围\n3. 点击开始处理\n4. 下载结果 & 查看光谱")
    
    # ========== 主区域：文件上传 ==========
    uploaded_file = st.file_uploader("📁 上传 TXT 数据文件", type=['txt'], 
                                     help="支持拖拽文件到此区域")
    
    # 检测新文件上传
    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.df = None
            st.session_state.df_results = None
            st.session_state.current_fig = None
            st.session_state.processed = False
            st.rerun()
        
        st.success(f"✅ 文件已加载：{uploaded_file.name}")
        
        # 数据预览
        with st.expander("📊 查看数据预览", expanded=False):
            if st.session_state.df is None:
                df_preview = txt_to_csv_content(uploaded_file.getvalue())
                if df_preview is not None:
                    st.session_state.df = df_preview
                    st.dataframe(df_preview.head(10))
                    st.write(f"共 {len(df_preview)} 行数据")
            else:
                st.dataframe(st.session_state.df.head(10))
                st.write(f"共 {len(st.session_state.df)} 行数据")
        
        # 处理按钮
        if st.button("🚀 开始分析", type="primary", key="process_btn"):
            st.session_state.d_range = (d_start, d_end)
            st.session_state.g_range = (g_start, g_end)
            
            if st.session_state.df is None:
                st.session_state.df = txt_to_csv_content(uploaded_file.getvalue())
            
            if st.session_state.df is not None:
                df_results = process_raman_data(st.session_state.df, 
                                                st.session_state.d_range, 
                                                st.session_state.g_range)
                
                if df_results is not None:
                    st.session_state.df_results = df_results
                    st.session_state.current_fig = None
                    # 初始化选择坐标为第一个点
                    st.session_state.selected_x = float(df_results['x'].iloc[0])
                    st.session_state.selected_y = float(df_results['y'].iloc[0])
                    st.session_state.processed = True
                    st.success("✅ 数据处理完成！请查看下方结果和下载按钮。")
                    st.rerun()
    
    # ========== 显示结果区域（始终显示，如果有结果）==========
    if st.session_state.df_results is not None:
        st.markdown("---")
        st.subheader("📈 分析结果")
        
        # 统计信息
        col1, col2 = st.columns(2)
        with col1:
            max_d = st.session_state.df_results.loc[
                st.session_state.df_results['D_Area'].idxmax()]
            st.metric("最大 D 峰面积", f"{max_d['D_Area']:.2f}", 
                     f"坐标：({max_d['x']}, {max_d['y']})")
        with col2:
            max_g = st.session_state.df_results.loc[
                st.session_state.df_results['G_Area'].idxmax()]
            st.metric("最大 G 峰面积", f"{max_g['G_Area']:.2f}", 
                     f"坐标：({max_g['x']}, {max_g['y']})")
        
        # 下载按钮（始终显示）
        csv = st.session_state.df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载结果 CSV",
            data=csv,
            file_name=f'raman_results_{st.session_state.uploaded_file_name.replace(".txt", "")}.csv',
            mime='text/csv',
            key="download_btn"
        )
        
        # 结果表格
        with st.expander("📋 查看完整结果表格", expanded=False):
            st.dataframe(st.session_state.df_results)
        
        # ========== 光谱查看器 ==========
        st.markdown("---")
        st.subheader("🎨 光谱查看器")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("**选择坐标点：**")
            
            # 坐标输入
            st.number_input("X 坐标", 
                           value=st.session_state.selected_x,
                           key="x_coord_input",
                           help="输入或选择 X 坐标")
            st.number_input("Y 坐标", 
                           value=st.session_state.selected_y,
                           key="y_coord_input",
                           help="输入或选择 Y 坐标")
            
            # 同步输入框值到 session_state
            st.session_state.selected_x = st.session_state.x_coord_input
            st.session_state.selected_y = st.session_state.y_coord_input
            
            # 快速选择按钮
            st.markdown("**快速选择：**")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.button("最大 D 峰", key="btn_max_d", on_click=select_max_d,
                         help="快速定位到 D 峰面积最大的坐标点")
            with col_btn2:
                st.button("最大 G 峰", key="btn_max_g", on_click=select_max_g,
                         help="快速定位到 G 峰面积最大的坐标点")
            
            # 绘制按钮
            st.markdown("---")
            if st.button("🎨 绘制光谱", type="secondary", key="plot_btn",
                        help="根据当前坐标绘制光谱图"):
                fig = plot_spectrum(st.session_state.df, 
                                   st.session_state.selected_x, 
                                   st.session_state.selected_y,
                                   st.session_state.d_range, 
                                   st.session_state.g_range)
                if fig:
                    st.session_state.current_fig = fig
                    st.success(f"已绘制坐标 ({st.session_state.selected_x}, {st.session_state.selected_y}) 的光谱")
        
        with col2:
            if st.session_state.current_fig is not None:
                st.pyplot(st.session_state.current_fig)
            else:
                st.info("👈 选择坐标后点击'绘制光谱'按钮")
    
    # ========== 底部说明 ==========
    st.markdown("---")
    st.markdown("""
    **提示：**
    - ✅ 结果和下载按钮会一直保留，即使绘制了新的光谱图
    - ✅ 可以随时修改参数重新处理
    - ✅ 上传新文件会自动重置所有数据
    - ✅ 快速选择按钮可快速定位最大峰坐标
    """)


if __name__ == "__main__":
    main()
