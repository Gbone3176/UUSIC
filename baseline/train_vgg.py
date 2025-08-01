import torch
import torch.nn as nn
import time
import threading
import numpy as np
from datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class GPUOccupier:
    def __init__(self, target_memory_gb=20, target_utilization=0.6):
        """
        GPU占用程序
        
        Args:
            target_memory_gb: 目标显存占用（GB）
            target_utilization: 目标利用率（0-1之间）
        """
        self.target_memory_gb = target_memory_gb
        self.target_utilization = target_utilization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.running = False
        self.memory_tensors = []
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法占用GPU")
            return
            
        print(f"🎯 目标: 占用 {target_memory_gb}GB 显存, 利用率 {target_utilization*100}%")
        print(f"📱 使用设备: {torch.cuda.get_device_name(0)}")
        
    def occupy_memory(self):
        """占用指定大小的显存"""
        print("💾 开始占用显存...")
        
        # 计算需要分配的显存大小（字节）
        target_bytes = self.target_memory_gb * 1024**3
        
        # 分块分配显存，避免一次性分配过大
        chunk_size = 1024**3  # 1GB per chunk
        num_chunks = int(target_bytes // chunk_size)
        remaining_bytes = target_bytes % chunk_size
        
        try:
            # 分配整数GB的块
            for i in range(num_chunks):
                chunk = torch.randn(chunk_size // 4, device=self.device, dtype=torch.float32)
                self.memory_tensors.append(chunk)
                current_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"  已分配: {current_memory:.2f}GB")
                
            # 分配剩余的显存
            if remaining_bytes > 0:
                chunk = torch.randn(remaining_bytes // 4, device=self.device, dtype=torch.float32)
                self.memory_tensors.append(chunk)
                
            final_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"✅ 显存占用完成: {final_memory:.2f}GB")
            
        except RuntimeError as e:
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"⚠️  显存分配受限: {current_memory:.2f}GB (错误: {e})")
    
    def create_dummy_model(self):
        """创建一个虚拟模型用于产生计算负载"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                )
                
            def forward(self, x):
                return self.layers(x)
        
        return DummyModel().to(self.device)
    
    def gpu_computation_worker(self):
        """GPU计算工作线程"""
        model = self.create_dummy_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 计算工作和休息时间
        work_time = self.target_utilization  # 工作时间比例
        rest_time = 1 - self.target_utilization  # 休息时间比例
        cycle_duration = 1.0  # 每个周期1秒
        
        work_duration = cycle_duration * work_time
        rest_duration = cycle_duration * rest_time
        
        print(f"🔄 开始计算负载: 工作{work_duration:.2f}s, 休息{rest_duration:.2f}s")
        
        iteration = 0
        while self.running:
            cycle_start = time.time()
            
            # 工作阶段
            work_start = time.time()
            while time.time() - work_start < work_duration and self.running:
                # 生成随机输入
                batch_size = 256
                input_data = torch.randn(batch_size, 1024, device=self.device)
                target = torch.randn(batch_size, 512, device=self.device)
                
                # 前向传播
                output = model(input_data)
                loss = nn.MSELoss()(output, target)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                iteration += 1
                if iteration % 10000 == 0:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    # print(f"[{current_time}] 迭代: {iteration}, 显存: {memory_used:.2f}GB, 损失: {loss.item():.4f}")
            
            # 休息阶段
            if rest_duration > 0 and self.running:
                time.sleep(rest_duration)
    
    def start(self):
        """开始占用GPU"""
        if not torch.cuda.is_available():
            return
            
        print("🚀 启动GPU占用程序...")
        print("-" * 60)
        
        # 先占用显存
        self.occupy_memory()
        
        # 启动计算线程
        self.running = True
        self.compute_thread = threading.Thread(target=self.gpu_computation_worker)
        self.compute_thread.daemon = True
        self.compute_thread.start()
        
        print("✅ GPU占用程序已启动")
        print("💡 按 Ctrl+C 退出程序")
        print("-" * 60)
        
        try:
            # 主线程保持运行，定期显示状态
            while True:
                time.sleep(10)
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_percent = (memory_used / memory_total) * 100
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] 显存使用: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_percent:.1f}%)")
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """停止占用GPU"""
        print("\n🛑 停止GPU占用...")
        self.running = False
        
        # 释放显存
        self.memory_tensors.clear()
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"✅ 已释放显存，当前占用: {final_memory:.2f}GB")
        print("👋 程序已退出")

def main():
    """主函数"""
    print("=" * 60)
    print("🎮 GPU占卡程序 v1.0")
    print("=" * 60)
    
    # 显示GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔍 检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({total_memory:.1f}GB)")
    else:
        print("❌ 未检测到CUDA设备")
        return
    
    print("-" * 60)
    
    # 创建并启动占卡程序
    occupier = GPUOccupier(target_memory_gb=25, target_utilization=0.6)
    occupier.start()

if __name__ == "__main__":
    main()
