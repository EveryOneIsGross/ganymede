# ganymede

Uses OLLAMA and QWEN2s smollest model for running local.

```
Problem: What is 2 * 16?
Model: qwen2:0.5b-instruct-fp16
Number of rollouts: 10

Rollout 1/10
Rollout 2/10
Rollout 3/10
Rollout 4/10
Rollout 5/10
Rollout 6/10
Rollout 7/10
Rollout 8/10
Rollout 9/10
Rollout 10/10

Best answer: The result of \(2\) times \(16\) is \(32\).
```

![dithered_ganymede](https://github.com/EveryOneIsGross/ganymede/assets/23621140/8d30c260-1ddf-4b1b-9680-149d63363170)

my budget smol implementation of q* : 
[https://arxiv.org/abs/2406.07394](https://arxiv.org/abs/2406.07394)

Just supply the problem in a .txt and the number of rollouts. 

```commandline
python ganymede.py --model "qwen2:0.5b-instruct-fp16" --rollouts 10 --problem_file problem.txt

```

```mermaid
graph TD
    A[Start] --> B[Parse Arguments]
    B --> C[Initialize LLMApi]
    C --> D[Create MCTSr Object]
    D --> E[Run MCTS]
    E --> F{For each rollout}
    F --> G[Select Node]
    G --> H[Expand Node]
    H --> I[Evaluate Node]
    I --> J[Backpropagate]
    J --> K{Should Terminate?}
    K -->|No| F
    K -->|Yes| L[Find Best Answer]
    L --> M[Print Best Answer]
    M --> N[End]

    subgraph "LLM Interactions"
    O[Get Feedback]
    P[Refine Answer]
    Q[Get Reward]
    end

    H --> O
    O --> P
    P --> H
    I --> Q
    Q --> I
```
