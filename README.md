# 基於深度強化學習的自主戰鬥機導航與戰鬥系統

## 1. 引言
在現代戰爭中，智能化無人駕駛系統的發展為軍事作戰提供了新的視角與可能性。自主戰鬥機作為這一領域的核心技術之一，擁有不僅能完成導航任務，還能在複雜的戰鬥環境中進行敵我識別、目標追蹤與精確打擊的能力。深度強化學習（Deep Reinforcement Learning, DRL）作為強大的機器學習技術，已在無人機導航、機器人控制等領域取得顯著進展。

本研究旨在利用深度強化學習設計並實現一個自主戰鬥機導航與戰鬥系統。通過構建基於 gymnasium 框架的模擬環境，我們使用深度Q網絡（Deep Q-Network, DQN）算法對智能體進行訓練，使其能夠在模擬環境中學習如何有效地執行戰鬥機的導航、目標追蹤及敵我交戰等任務。

## 2. 貢獻
本研究的主要貢獻如下：

- **環境設計與構建**：基於 gymnasium 框架，設計了一個仿真環境，模擬了自主戰鬥機的運動學、目標追蹤、敵人交戰等複雜行為。該環境的設計能夠提供多樣化的訓練場景，促使智能體學習到高效的策略。
  
- **深度強化學習算法應用**：運用穩定基線3（Stable-Baselines3）中的 DQN 算法，訓練了自主戰鬥機智能體，實現了目標追蹤、躲避敵人攻擊及進行有效攻擊的策略學習。
  
- **訓練過程中的優化與控制**：為避免過度擬合，提出了早停機制（Early Stopping），並通過監控獎勳變化自動停止訓練，從而提升訓練效率與穩定性。
  
- **可視化與結果分析**：使用 TensorBoard 和 Matplotlib 等工具對訓練過程中的主要指標（例如獎勳、探索率、Q 值等）進行可視化，提供了對模型訓練過程及結果的深入分析。

## 3. 系統架構
本系統的設計可分為以下幾個模塊：

### 3.1 環境設計
環境是基於 gymnasium 框架構建的，模擬了戰鬥機的運動學、目標追蹤以及敵我交戰的行為。環境的設計包括以下幾個方面：

- **狀態空間**：環境中的每一個狀態包含了戰鬥機的位置、速度、飛行方向、與目標和敵人之間的距離等多維信息。這些數據反映了當前戰鬥機的狀態，幫助智能體做出決策。

- **動作空間**：動作空間包括了6個離散動作：加速、減速、轉向（左、右、上、下）以及射擊。這些動作的設計允許智能體根據當前狀況做出適應性的控制。

- **獎勳設計**：基於智能體的行為來設計獎勳函數，包括：
  - 目標追蹤獎勳：當戰鬥機接近目標時，智能體獲得正向獎勳；
  - 敵人擊中獎勳：成功擊中敵人後給予較高的獎勳；
  - 避免被擊中獎勳：成功躲避敵人攻擊時給予獎勳；
  - 懲罰：在不必要的時候進行加速或是無效的射擊，會使智能體受到懲罰。

### 3.2 深度強化學習模型
我們使用 Stable-Baselines3 框架中的 DQN 算法來訓練智能體，並設計了如下的網絡結構與訓練參數：

- **網絡結構**：我們使用了三層隱藏層的神經網絡，每層包含 256 個神經元。這個結構能夠有效地捕捉複雜的狀態空間和動作之間的關聯。

- **超參數設置**：
  - 學習率：5e-5，這是經過多次調整後選定的最優學習率；
  - 批量大小：256，每次訓練時從經驗池中選取256個樣本；
  - 緩衝區大小：500000，足以存儲大量的過去經驗；
  - 折扣因子：0.99，這表明未來的回報對當前行動的影響；
  - 探索率：從 1.0 線性衰減到 0.1，促使智能體逐漸從隨機探索轉向依賴學到的策略。

### 3.3 回調函數
為了提升訓練效率並防止過度訓練，我們在訓練過程中引入了兩個重要的回調函數：

- **TensorboardCallback**：用於記錄訓練過程中的關鍵指標（如獎勳、Q 值、探索率等），並通過 TensorBoard 進行可視化，幫助我們深入了解訓練過程。
  
- **EarlyStoppingCallback**：這個回調函數監控獎勳的變化，若在若干步驟內無法提升獎勳，則提前終止訓練，以防止過度擬合並節省計算資源。

## 4. 代碼實現

### 4.1 環境設計
```python
class FighterJetEnv(gym.Env):
    def __init__(self):
        # 初始化環境參數
        self.width = 800
        self.height = 800
        self.max_steps = 2000
        self.bullet_speed = 7
        self.targeting_zone_radius = 200
        self.enemy_observation_radius = 250
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self):
        # 重置環境
        self.jet_pos = np.array([self.width // 2, self.height - 50], dtype=np.float32)
        self.target_pos = np.array([random.randint(50, self.width-50), random.randint(50, self.height-50)], dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        # 根據動作更新環境狀態
        reward = self._calculate_reward()
        done = self._check_done()
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # 返回觀測值
        return np.array([self.jet_pos[0], self.jet_pos[1], self.target_pos[0], self.target_pos[1], ...], dtype=np.float32)
