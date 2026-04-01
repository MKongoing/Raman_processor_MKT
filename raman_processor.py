import csv
import numpy as np
import pandas as pd
import os
from scipy.integrate import trapz
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def txt_to_csv(txt_file, csv_file=None):
    """
    将特殊格式的txt文件转换为csv文件，并返回转换后的文件路径
    输出文件默认保存在输入文件同一目录下

    参数:
    - txt_file: 输入的txt文件路径
    - csv_file: (可选)输出的csv文件路径，默认自动生成

    返回:
    - 转换后的CSV文件完整路径
    """
    try:
        # 获取输入文件所在目录
        input_dir = os.path.dirname(txt_file)

        # 自动生成输出文件名（同目录下）
        if csv_file is None:
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            csv_file = os.path.join(input_dir, f"{base_name}_converted.csv")

        # 打开txt文件并读取数据
        with open(txt_file, 'r', encoding='utf-8') as txt:
            lines = [line for line in txt.readlines() if line.strip()]  # 过滤空行

        if not lines:
            raise ValueError("输入文件为空")

        # 解析标题行（第一个非空行），按两个制表符拆分
        header = lines[0].strip().split('\t\t')

        # 解析数据行，按单个制表符拆分
        data = []
        for line in lines[1:]:
            # 处理可能存在的混合分隔符情况
            cleaned_line = line.replace('\t\t', '\t').strip()
            if cleaned_line:
                data.append(cleaned_line.split('\t'))

        # 写入csv文件
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data)

        print(f"成功将 {txt_file} 转换为 {csv_file}")
        return csv_file

    except Exception as e:
        print(f"转换失败: {e}")
        return None


def get_valid_filepath(prompt="请输入数据文件路径（支持拖放文件至此）：", file_type=None):
    """获取有效的文件路径"""
    while True:
        file_path = input(prompt).strip('"')
        if not os.path.exists(file_path):
            print(f"错误：文件 '{file_path}' 不存在")
            continue

        if file_type and not file_path.lower().endswith(file_type.lower()):
            print(f"错误：需要 {file_type} 格式文件")
            continue

        return file_path


def correct_baseline(wave, intensity, deg=1):
    """使用多项式拟合进行基线校正"""
    if len(wave) < 2:
        return intensity, np.zeros_like(intensity)
    p = np.polyfit(wave, intensity, deg=deg)
    baseline = np.polyval(p, wave)
    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)
    return corrected, baseline


def process_raman_data(csv_file):
    """处理拉曼光谱数据，结果文件保存在输入文件同目录"""
    try:
        # 获取输入文件所在目录
        input_dir = os.path.dirname(csv_file)

        # 读取数据
        df = pd.read_csv(
            csv_file,
            names=["x", "y", "Wave", "Intensity"],
            skiprows=1,
            comment='#',
            dtype={"x": float, "y": float, "Wave": float, "Intensity": float},
            na_values=["---", "nan"]
        ).dropna()

        # 定义峰范围
        d_range = (1310, 1410)
        g_range = (1520, 1620)

        # 处理每个坐标点
        xy_coords = df[['x', 'y']].drop_duplicates().values
        results = []

        for x, y in xy_coords:
            spectrum = df[(df['x'] == x) & (df['y'] == y)]
            wave = spectrum['Wave'].values
            intensity = spectrum['Intensity'].values

            # 确保波长升序
            if wave[0] > wave[-1]:
                wave = wave[::-1]
                intensity = intensity[::-1]

            # 平滑处理
            if len(intensity) >= 11:
                intensity = savgol_filter(intensity, window_length=11, polyorder=2)
            else:
                print(f"警告：坐标 ({x}, {y}) 数据过短，跳过平滑处理")

            # 截断负值
            intensity = np.maximum(intensity, 0)

            # 计算D峰积分
            d_mask = (wave >= d_range[0]) & (wave <= d_range[1])
            wave_d = wave[d_mask]
            intensity_d = intensity[d_mask]
            corrected_d, _ = correct_baseline(wave_d, intensity_d)
            d_area = trapz(corrected_d, wave_d)

            # 计算G峰积分
            g_mask = (wave >= g_range[0]) & (wave <= g_range[1])
            wave_g = wave[g_mask]
            intensity_g = intensity[g_mask]
            corrected_g, _ = correct_baseline(wave_g, intensity_g)
            g_area = trapz(corrected_g, wave_g)

            results.append([x, y, d_area, g_area])

        # 保存结果到同目录
        output_filename = os.path.join(input_dir, f"raman_results_{os.path.basename(csv_file)}")
        df_results = pd.DataFrame(results, columns=["x", "y", "D_area", "G_area"])
        df_results.to_csv(output_filename, index=False)

        # 查找最大值坐标
        max_d_row = df_results.loc[df_results['D_area'].idxmax()]
        max_g_row = df_results.loc[df_results['G_area'].idxmax()]

        print(f"\n分析完成，结果已保存至 {output_filename}")
        print(f"最大D峰积分坐标点：x={max_d_row['x']:.2f}, y={max_d_row['y']:.2f}，面积为{max_d_row['D_area']:.2f}")
        print(f"最大G峰积分坐标点：x={max_g_row['x']:.2f}, y={max_g_row['y']:.2f}，面积为{max_g_row['G_area']:.2f}")

        return df, df_results

    except Exception as e:
        print(f"数据处理失败: {e}")
        return None, None


def plot_spectrum(df, x, y):
    """绘制指定坐标的光谱图"""
    try:
        spectrum = df[(df['x'] == x) & (df['y'] == y)]
        if spectrum.empty:
            print(f"未找到坐标 ({x}, {y}) 的数据")
            return False

        wave = spectrum['Wave'].values
        intensity = spectrum['Intensity'].values

        # 确保波长升序
        if wave[0] > wave[-1]:
            wave = wave[::-1]
            intensity = intensity[::-1]

        # 平滑处理
        if len(intensity) >= 11:
            intensity = savgol_filter(intensity, window_length=11, polyorder=2)

        # 截断负值
        intensity = np.maximum(intensity, 0)

        # 定义峰范围
        d_range = (1310, 1410)
        g_range = (1520, 1620)

        # 创建图形
        plt.figure(figsize=(12, 6))
        plt.plot(wave, intensity, label="Original Smoothed", color="gray", alpha=0.5)

        # 处理D峰
        d_mask = (wave >= d_range[0]) & (wave <= d_range[1])
        wave_d = wave[d_mask]
        intensity_d = intensity[d_mask]
        corrected_d, baseline_d = correct_baseline(wave_d, intensity_d)
        plt.fill_between(wave_d, corrected_d, color='red', alpha=0.3,
                         label=f'D Peak (Area: {trapz(corrected_d, wave_d):.2f})')
        plt.plot(wave_d, baseline_d, 'r--', label='D Baseline')

        # 处理G峰
        g_mask = (wave >= g_range[0]) & (wave <= g_range[1])
        wave_g = wave[g_mask]
        intensity_g = intensity[g_mask]
        corrected_g, baseline_g = correct_baseline(wave_g, intensity_g)
        plt.fill_between(wave_g, corrected_g, color='green', alpha=0.3,
                         label=f'G Peak (Area: {trapz(corrected_g, wave_g):.2f})')
        plt.plot(wave_g, baseline_g, 'g--', label='G Baseline')

        plt.title(f"Raman Spectrum at ({x}, {y})")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()
        return True

    except Exception as e:
        print(f"绘图失败: {e}")
        return False


def main():
    """主程序"""
    print("=" * 50)
    print("拉曼光谱数据处理系统")
    print("=" * 50)

    # 第一步：文件格式转换
    print("\n[第一步] TXT文件转换")
    txt_file = get_valid_filepath("请输入TXT文件路径（支持拖放）：", ".txt")
    csv_file = txt_to_csv(txt_file)

    if not csv_file:
        return

    # 第二步：数据处理
    print("\n[第二步] 拉曼数据分析")
    df, df_results = process_raman_data(csv_file)

    if df is None:
        return

    # 第三步：交互式绘图
    print("\n[第三步] 光谱可视化")
    while True:
        try:
            user_input = input("\n请输入x坐标（输入q退出）：")
            if user_input.lower() == 'q':
                break

            user_x = float(user_input)
            user_y = float(input("请输入y坐标："))

            plot_spectrum(df, user_x, user_y)

            continue_plot = input("是否继续查看其他坐标？(y/n): ").lower()
            if continue_plot != 'y':
                break

        except ValueError:
            print("输入无效，请重新输入坐标或输入q退出")


if __name__ == "__main__":
    main()