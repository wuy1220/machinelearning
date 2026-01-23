import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import os
import sys
import json

def main():
    print("=" * 80)
    print("Data Generator Comprehensive Validation Program")
    print("=" * 80)
    
    # ========================================================================
    # 导入测试模块
    # ========================================================================
    try:
        from mdof_improved_gvr_2 import (
            ImprovedJacketPlatformSimulator,
            TimeStackedGVRFeatureExtractor,
            ImprovedDamageDataGenerator
        )
        print("[OK] Successfully imported data generator module")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        print("Please ensure mdof_improved_gvr_2.py is in the current directory")
        return False
    
    output_dir = './validation_output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    test_results = {'passed': 0, 'failed': 0, 'details': []}
    
    # ========================================================================
    # 测试日志记录函数
    # ========================================================================
    def log_test(test_name, passed, message=""):
        nonlocal test_results
        if passed:
            test_results['passed'] += 1
            status = "[PASS]"
            print(f"{status} {test_name}")
            if message:
                print(f"       {message}")
        else:
            test_results['failed'] += 1
            status = "[FAIL]"
            print(f"{status} {test_name}")
            if message:
                print(f"       Error: {message}")
        
        test_results['details'].append({
            'name': test_name,
            'passed': passed,
            'message': message
        })
    
    # ========================================================================
    # 测试组 1: 物理与动力学验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test Group 1: Physics & Dynamics Validation")
    print("=" * 80)
    
    # 测试 1.1: 固有频率合理性
    try:
        sim = ImprovedJacketPlatformSimulator(
            num_degrees=10,
            dt=0.005,
            duration=10.0,
            damping_ratio=0.05,
            seed=42
        )
        
        freqs = sim.natural_frequencies
        all_positive = np.all(freqs > 0)
        monotonically_increasing = np.all(freqs[1:] > freqs[:-1])
        freq_hz = freqs[0] / (2 * np.pi)
        freq_in_range = 0.1 < freq_hz < 2.0
        
        log_test("Natural frequencies are positive", all_positive, 
                 f"Range: {freqs.min():.4f} - {freqs.max():.4f} rad/s")
        log_test("Natural frequencies monotonically increase", monotonically_increasing)
        log_test("First frequency in reasonable range (0.1-2 Hz)", freq_in_range,
                 f"First frequency: {freq_hz:.4f} Hz")
        
    except Exception as e:
        log_test("Natural frequency test", False, str(e))
    
    # 测试 1.2: 损伤对模态的影响
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=10, dt=0.005, duration=10.0, seed=42)
        freq_healthy = sim.natural_frequencies.copy()
        K_damaged = sim.apply_damage(damaged_dofs=[5], severity_ratios=[0.5])
        
        eigenvalues_damaged, _ = np.linalg.eigh(np.linalg.inv(sim.M) @ K_damaged)
        freq_damaged = np.sqrt(np.maximum(eigenvalues_damaged, 1e-10))
        freq_decreased = freq_damaged[0] < freq_healthy[0]
        
        log_test("Damage reduces frequency", freq_decreased,
                 f"Healthy: {freq_healthy[0]/(2*np.pi):.4f} Hz -> Damaged: {freq_damaged[0]/(2*np.pi):.4f} Hz")
        
        # 可视化频率对比
        fig, ax = plt.subplots(figsize=(8, 5))
        modes = np.arange(1, len(freq_healthy) + 1)
        ax.plot(modes, freq_healthy/(2*np.pi), 'b-o', label='Healthy', linewidth=2)
        ax.plot(modes, freq_damaged/(2*np.pi), 'r-s', label='Damaged', linewidth=2)
        ax.set_xlabel('Mode Number')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Effect of Damage on Natural Frequencies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_comparison.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        log_test("Damage effect test", False, str(e))
    
    # 测试 1.3: 刚度矩阵结构
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=5, dt=0.005, duration=10.0, seed=42)
        K = sim.K
        n = sim.num_degrees
        
        is_square = K.shape == (n, n)
        is_symmetric = np.allclose(K, K.T)
        eigenvals = np.linalg.eigvals(K)
        is_positive_definite = np.all(eigenvals > 0)
        
        is_tridiagonal = True
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(K[i, j]) > 1e-10:
                    is_tridiagonal = False
                    break
        
        log_test("Stiffness matrix is square", is_square)
        log_test("Stiffness matrix is symmetric", is_symmetric)
        log_test("Stiffness matrix is positive definite", is_positive_definite)
        log_test("Stiffness matrix has tridiagonal structure", is_tridiagonal)
        
    except Exception as e:
        log_test("Stiffness matrix structure test", False, str(e))
    
    # ========================================================================
    # 测试组 2: 数值计算与求解器验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test Group 2: Numerical & Solver Validation")
    print("=" * 80)
    
    # 测试 2.1: 零输入零状态响应
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=5, dt=0.001, duration=1.0, damping_ratio=0.0, seed=42)
        sim.C = np.zeros_like(sim.C)
        F_zero = np.zeros((sim.num_steps, sim.num_degrees))
        resp = sim.simulate_response(sim.K, F_zero)
        
        max_response = np.max(np.abs(resp))
        is_zero_response = max_response < 1e-8
        log_test("Zero input produces near-zero response", is_zero_response,
                 f"Max response: {max_response:.2e}")
        
    except Exception as e:
        log_test("Zero input test", False, str(e))
    
    # 测试 2.2: 能量守恒（稳定性）
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=3, dt=0.001, duration=2.0, damping_ratio=0.0, seed=42)
        sim.C = np.zeros_like(sim.C)
        F = np.zeros((sim.num_steps, sim.num_degrees))
        F[0, 0] = 1000.0
        resp = sim.simulate_response(sim.K, F)
        
        max_response = np.max(np.abs(resp))
        is_stable = np.isfinite(max_response) and max_response < 1e6
        has_no_nan = not np.any(np.isnan(resp))
        has_no_inf = not np.any(np.isinf(resp))
        
        log_test("Response is numerically stable", is_stable)
        log_test("Response has no NaN values", has_no_nan)
        log_test("Response has no Inf values", has_no_inf)
        
    except Exception as e:
        log_test("Energy conservation test", False, str(e))
    
    # 测试 2.3: 激励信号滤波
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=10, dt=0.005, duration=10.0, seed=42)
        F = sim.generate_excitation(excitation_type='filtered_noise', amplitude=1.0)
        
        n_fft = 2**14
        fft_vals = np.fft.fft(F[:, 0], n=n_fft)
        fft_freq = np.fft.fftfreq(n_fft, sim.dt)
        
        low_freq_idx = np.argmin(np.abs(fft_freq - 0.2))
        high_freq_idx = np.argmin(np.abs(fft_freq - 3.0))
        
        power_spectrum = np.abs(fft_vals) ** 2
        total_power = np.sum(power_spectrum)
        band_power = np.sum(power_spectrum[low_freq_idx:high_freq_idx])
        
        power_ratio = band_power / total_power
        is_filtered = power_ratio > 0.5
        log_test("Excitation signal is properly filtered", is_filtered,
                 f"Band power ratio: {power_ratio*100:.1f}%")
        
        # 可视化频谱
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(fft_freq[:n_fft//2], 10*np.log10(power_spectrum[:n_fft//2]))
        ax.axvline(0.2, color='r', linestyle='--', label='Low freq cutoff')
        ax.axvline(3.0, color='r', linestyle='--', label='High freq cutoff')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB)')
        ax.set_title('Excitation Signal Spectrum Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'excitation_spectrum.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        log_test("Excitation filtering test", False, str(e))
    
    # ========================================================================
    # 测试组 3: 特征提取逻辑验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test Group 3: Feature Extraction Validation")
    print("=" * 80)
    
    # 测试 3.1: DI 计算
    try:
        dt = 0.005
        extractor = TimeStackedGVRFeatureExtractor(
            dt=dt, window_length=500, step_size=10,
            num_stack_windows=50, cutoff_freq=5.0
        )
        
        n_steps = 5000
        n_channels = 10
        healthy_signal = np.random.randn(n_steps, n_channels) * 0.1
        damaged_signal = healthy_signal.copy()
        damaged_signal[1000:4000, 5] += 2.0  # 添加损伤特征
        
        features = extractor.extract_gvr_features(damaged_signal, healthy_signal)
        DI_series = features['DI']
        
        correct_shape = DI_series.shape[1] == n_channels
        di_channel_5 = DI_series[:, 5].mean()
        di_other_channels = np.delete(DI_series, 5, axis=1).mean()
        damage_detected = di_channel_5 > di_other_channels * 2
        
        log_test("DI calculation is correct", correct_shape and damage_detected,
                 f"Damaged channel DI: {di_channel_5:.4f}, Others: {di_other_channels:.4f}")
        
    except Exception as e:
        log_test("DI calculation test", False, str(e))
    
    # 测试 3.2: GVR 计算
    try:
        dt = 0.005
        extractor = TimeStackedGVRFeatureExtractor(
            dt=dt, window_length=500, step_size=10,
            num_stack_windows=50, cutoff_freq=5.0
        )
        
        DI_series = np.zeros((100, 10))
        DI_series[30:50, 3] = 0.5  # 人为添加突变
        
        DI_prime, DI_double_prime = extractor.compute_gvr(DI_series)
        
        correct_shape = (DI_prime.shape == DI_series.shape and 
                        DI_double_prime.shape == DI_series.shape)
        
        spike_detected = DI_double_prime[30:50, 3].sum() > DI_double_prime[:30, 3].sum() * 2
        
        log_test("GVR calculation is correct", correct_shape and spike_detected,
                 "Spike feature detected")
        
    except Exception as e:
        log_test("GVR calculation test", False, str(e))
    
    # 测试 3.3: 特征图生成
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=10, dt=0.005, duration=30.0, seed=42)
        gvr_extractor = TimeStackedGVRFeatureExtractor(
            dt=sim.dt, window_length=500, step_size=10,
            num_stack_windows=224, cutoff_freq=2.0
        )
        
        F = sim.generate_excitation(excitation_type='filtered_noise')
        healthy = sim.simulate_response(sim.K0, F)
        K_damaged = sim.apply_damage(damaged_dofs=[5], severity_ratios=[0.4])
        damaged = sim.simulate_response(K_damaged, F)
        
        stacked_features, num_samples = gvr_extractor.extract_stacked_gvr_features(damaged, healthy)
        feature_maps = gvr_extractor.generate_time_space_feature_map(stacked_features)
        
        correct_shape = feature_maps.shape[1:] == (224, 224, 3)
        in_range = (feature_maps.min() >= 0.0 and feature_maps.max() <= 1.0)
        no_nan = not np.any(np.isnan(feature_maps))
        no_inf = not np.any(np.isinf(feature_maps))
        
        log_test("Feature map shape is correct (224, 224, 3)", correct_shape,
                 f"Actual shape: {feature_maps.shape}")
        log_test("Feature maps normalized to [0, 1]", in_range)
        log_test("Feature maps have no NaN", no_nan)
        log_test("Feature maps have no Inf", no_inf)
        
        # 可视化特征图
        if feature_maps.shape[0] > 0:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            img = feature_maps[0]
            axes[0].imshow(img[:,:,0], cmap='hot', aspect='auto')
            axes[0].set_title("R Channel (DI')")
            axes[0].set_xlabel('Sensor Position')
            axes[0].set_ylabel('Time')
            
            axes[1].imshow(img[:,:,1], cmap='hot', aspect='auto')
            axes[1].set_title("G Channel (DI'')")
            axes[1].set_xlabel('Sensor Position')
            axes[1].set_yticks([])
            
            axes[2].imshow(img[:,:,2], cmap='hot', aspect='auto')
            axes[2].set_title('B Channel (DI)')
            axes[2].set_xlabel('Sensor Position')
            axes[2].set_yticks([])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_maps.png'), dpi=150)
            plt.close()
        
    except Exception as e:
        log_test("Feature map generation test", False, str(e))
    
    # ========================================================================
    # 测试组 4: 损伤机制逻辑验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test Group 4: Damage Mechanism Logic")
    print("=" * 80)
    
    try:
        sim = ImprovedJacketPlatformSimulator(num_degrees=5, dt=0.005, duration=10.0, seed=42)
        K0 = sim.K0.copy()
        
        K_damaged_single = sim.apply_damage(damaged_dofs=[2], severity_ratios=[0.5])
        
        k0_unchanged = np.allclose(K0, sim.K0)
        is_different = not np.allclose(K0, K_damaged_single)
        
        log_test("apply_damage does not modify original K0", k0_unchanged)
        log_test("apply_damage generates damaged matrix", is_different)
        
        K_damaged_multi = sim.apply_damage(damaged_dofs=[1, 3], severity_ratios=[0.3, 0.4])
        
        is_multi_different = not np.allclose(K0, K_damaged_multi)
        is_single_multi_different = not np.allclose(K_damaged_single, K_damaged_multi)
        
        log_test("Multi-damage generates different matrix", 
                 is_multi_different and is_single_multi_different)
        
    except Exception as e:
        log_test("Damage logic test", False, str(e))
    
    # ========================================================================
    # 测试组 5: 数据完整性验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("Test Group 5: Data Integrity Validation")
    print("=" * 80)
    
    # 测试 5.1: HDF5 读写
    try:
        test_dir = os.path.join(output_dir, 'test_data')
        os.makedirs(test_dir, exist_ok=True)
        
        sim = ImprovedJacketPlatformSimulator(num_degrees=10, dt=0.01, duration=10.0, seed=42)
        gvr_extractor = TimeStackedGVRFeatureExtractor(
            dt=sim.dt, window_length=200, step_size=5,
            num_stack_windows=50, cutoff_freq=2.0
        )
        
        generator = ImprovedDamageDataGenerator(
            simulator=sim, gvr_extractor=gvr_extractor, output_dir=test_dir
        )
        
        data = generator.generate_single_damage_scenario(
            damaged_dofs=[5], severity_ratios=[0.3], scenario_id=0, save_data=True
        )
        
        filename = os.path.join(test_dir, 'scenario_0000.h5')
        file_exists = os.path.exists(filename)
        
        if file_exists:
            with h5py.File(filename, 'r') as f:
                has_acceleration = 'acceleration' in f
                has_feature_maps = 'feature_maps' in f
                has_labels = 'labels' in f
                has_damage_class = 'damage_class' in f
                
                acc_loaded = f['acceleration'][:]
                feat_loaded = f['feature_maps'][:]
                labels_loaded = f['labels'][:]
                
                acc_dtype_correct = acc_loaded.dtype == np.float32
                feat_dtype_correct = feat_loaded.dtype == np.float32
                labels_dtype_correct = labels_loaded.dtype == np.uint8
                
                acc_shape_correct = len(acc_loaded.shape) == 2
                feat_shape_correct = feat_loaded.shape[1:] == (50, 224, 3)
                labels_shape_correct = labels_loaded.shape == (10,)
            
            log_test("HDF5 file created successfully", file_exists)
            log_test("Contains acceleration data", has_acceleration)
            log_test("Contains feature maps", has_feature_maps)
            log_test("Contains labels", has_labels)
            log_test("Contains damage class", has_damage_class)
            log_test("Acceleration data type is float32", acc_dtype_correct)
            log_test("Feature maps data type is float32", feat_dtype_correct)
            log_test("Labels data type is uint8", labels_dtype_correct)
            log_test("Feature maps shape is correct", feat_shape_correct)
            
            metadata_file = os.path.join(test_dir, 'metadata.json')
            metadata_exists = os.path.exists(metadata_file)
            log_test("Metadata file generated", metadata_exists)
        else:
            log_test("HDF5 file creation", False, "File not generated")
            file_exists = False
        
    except Exception as e:
        log_test("HDF5 data test", False, str(e))
        file_exists = False
    
    # 测试 5.2: 端到端流程
    try:
        e2e_dir = os.path.join(output_dir, 'e2e_test')
        
        sim = ImprovedJacketPlatformSimulator(num_degrees=15, dt=0.005, duration=20.0, seed=42)
        gvr_extractor = TimeStackedGVRFeatureExtractor(
            dt=sim.dt, window_length=300, step_size=10,
            num_stack_windows=100, cutoff_freq=2.0
        )
        
        generator = ImprovedDamageDataGenerator(
            simulator=sim, gvr_extractor=gvr_extractor, output_dir=e2e_dir
        )
        
        generator.generate_comprehensive_dataset(
            num_scenarios=10, healthy_ratio=0.3,
            min_severity=0.2, max_severity=0.6
        )
        
        files = os.listdir(e2e_dir)
        h5_files = [f for f in files if f.endswith('.h5')]
        
        correct_file_count = len(h5_files) == 10
        metadata_exists = 'metadata.json' in files
        
        all_valid = True
        for h5_file in h5_files:
            filepath = os.path.join(e2e_dir, h5_file)
            try:
                with h5py.File(filepath, 'r') as f:
                    assert 'acceleration' in f
                    assert 'feature_maps' in f
                    assert 'labels' in f
                    assert 'damage_class' in f
            except:
                all_valid = False
                break
        
        log_test("End-to-end pipeline generates correct files", 
                 correct_file_count and metadata_exists and all_valid,
                 f"Generated {len(h5_files)} H5 files")
        
    except Exception as e:
        log_test("End-to-end pipeline test", False, str(e))
    
    # ========================================================================
    # 生成摘要报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("Validation Summary Report")
    print("=" * 80)
    
    total = test_results['passed'] + test_results['failed']
    pass_rate = (test_results['passed'] / total * 100) if total > 0 else 0
    
    print(f"\nTotal tests: {total}")
    print(f"Passed: {test_results['passed']} ({pass_rate:.1f}%)")
    print(f"Failed: {test_results['failed']}")
    
    print("\n" + "-" * 80)
    print("Detailed Results:")
    print("-" * 80)
    
    for i, detail in enumerate(test_results['details'], 1):
        status = "OK" if detail['passed'] else "FAIL"
        print(f"{i:2d}. [{status}] {detail['name']}")
        if detail['message']:
            print(f"    {detail['message']}")
    
    print("\n" + "=" * 80)
    
    # 保存报告到文件
    report_file = os.path.join(output_dir, 'validation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Data Generator Validation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tests: {total}\n")
        f.write(f"Passed: {test_results['passed']} ({pass_rate:.1f}%)\n")
        f.write(f"Failed: {test_results['failed']}\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        for i, detail in enumerate(test_results['details'], 1):
            status = "PASS" if detail['passed'] else "FAIL"
            f.write(f"{i:2d}. [{status}] {detail['name']}\n")
            if detail['message']:
                f.write(f"    {detail['message']}\n")
    
    print(f"Detailed report saved to: {report_file}")
    
    return pass_rate >= 80


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nValidation program error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
