"""
python ganymede.py --model "qwen2:0.5b-instruct-fp16" --rollouts 10 --problem_file problem.txt

"""

import json
import math
import random
from typing import List, Dict, Any
from openai import OpenAI
import re

class LLMApi:
    def __init__(self):
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required, but unused
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="qwen2:0.5b-instruct-fp16",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in LLM API call: {e}")
            return ""

class MCTSrNode:
    def __init__(self, problem: str, answer: str, parent=None):
        self.problem = problem
        self.answer = answer
        self.parent = parent
        self.children: List[MCTSrNode] = []
        self.visits = 0
        self.q_value = 0
        self.rewards: List[float] = []

class MCTSr:
    def __init__(self, llm_api: LLMApi, problem: str, max_rollouts: int):
        self.llm_api = llm_api
        self.problem = problem
        self.max_rollouts = max_rollouts
        self.root = self.initialize_root()
        self.output_data: Dict[str, Any] = {"problem": problem, "rollouts": []}
        self.c = 1  # UCB exploration parameter
        self.epsilon = 1e-6  # Small constant to avoid division by zero

    def initialize_root(self) -> MCTSrNode:
        naive_answer = self.llm_api.generate(f"Solve this problem: {self.problem}")
        dummy_answers = [
            "I don't know.",
            "I can't understand this question.",
            "I can't help with this question.",
            "I don't know how to solve this question.",
            "I don't know the answer to this question.",
            "I don't know the answer to this question, sorry."
        ]
        dummy_answer = random.choice(dummy_answers)
        
        root = MCTSrNode(self.problem, "")
        root.children = [MCTSrNode(self.problem, naive_answer, root), MCTSrNode(self.problem, dummy_answer, root)]
        return root

    def run(self) -> str:
        for rollout in range(self.max_rollouts):
            print(f"Rollout {rollout + 1}/{self.max_rollouts}")
            node = self.select()
            new_node = self.expand(node)
            reward = self.evaluate(new_node)
            if reward is None:
                print(f"Failed to extract score in rollout {rollout + 1}. Stopping early.")
                break
            self.backpropagate(new_node, reward)

            self.output_data["rollouts"].append({
                "rollout_number": rollout + 1,
                "selected_answer": node.answer,
                "refined_answer": new_node.answer,
                "reward": reward
            })

            if self.should_terminate():
                break

        best_answer = self.best_answer()
        self.output_data["best_answer"] = best_answer
        self.save_output()
        return best_answer
    
    def select(self) -> MCTSrNode:
        node = self.root
        while node.children:
            if len(node.children) < 2:  # If not fully expanded
                return node
            node = max(node.children, key=lambda n: self.ucb_value(n))
        return node

    def ucb_value(self, node: MCTSrNode) -> float:
        if node.visits == 0:
            return float('inf')
        exploitation = node.q_value
        exploration = math.sqrt(math.log(node.parent.visits) / (node.visits + self.epsilon))
        return exploitation + self.c * exploration

    def expand(self, node: MCTSrNode) -> MCTSrNode:
        feedback = self.get_feedback(node)
        refined_answer = self.refine_answer(node, feedback)
        new_node = MCTSrNode(node.problem, refined_answer, parent=node)
        node.children.append(new_node)
        return new_node

    def evaluate(self, node: MCTSrNode) -> float:
        reward = self.get_reward(node)
        if reward is None:
            print(f"Failed to get valid reward for node: {node.answer}")
            return 0.0  # Return a default value or handle this case as needed
        return reward

    def backpropagate(self, node: MCTSrNode, reward: float):
        while node:
            node.visits += 1
            node.rewards.append(reward)
            node.q_value = self.calculate_q_value(node)
            node = node.parent

    def calculate_q_value(self, node: MCTSrNode) -> float:
        if not node.rewards:
            return 0
        return 0.5 * (min(node.rewards) + sum(node.rewards) / len(node.rewards))

    def best_answer(self) -> str:
        return max(self.root.children, key=lambda n: n.q_value).answer

    def get_feedback(self, node: MCTSrNode) -> str:
        prompt = f"""
        Question: {node.problem}
        Current answer: {node.answer}
        
        Provide a reflection or feedback to correct this answer better. Analyze this answer strictly and critically, pointing out every flaw and possible imperfection to minimize the score.
        
        Let's think step by step.
        """
        return self.llm_api.generate(prompt)

    def refine_answer(self, node: MCTSrNode, feedback: str) -> str:
        prompt = f"""
        Question: {node.problem}
        Current answer: {node.answer}
        Feedback: {feedback}
        
        Please refine the answer according to the reflection or feedback. The response should begin with [reasoning process]...[Verification]... and end with "[Final Answer] The answer is [answer formula]"
        
        Let's think step by step.
        """
        return self.llm_api.generate(prompt)

    def get_reward(self, node: MCTSrNode) -> float:
        prompt = f"""
        Question: {node.problem}
        Answer: {node.answer}
        
        Analyze this answer strictly and critically, and point out every flaw for every possible imperfection to minimize the score. You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative.
        
        Output a score between -100 and +100 (e.g., from -100 to +100).
        
        Your response MUST end with the following format:
        [Score]: X
        Where X is the numerical score between -100 and +100.
        """
        max_retries = 3
        for _ in range(max_retries):
            response = self.llm_api.generate(prompt)
            score = self.extract_score(response)
            if score is not None:
                return min(score, 95)  # Implement full score suppression
        print(f"Failed to extract score after {max_retries} attempts. Last response: {response}")
        return None

    @staticmethod
    def extract_score(response: str) -> float:
        try:
            # Look for [Score]: X pattern
            score_match = re.search(r'\[Score\]: *(-?\d+(?:\.\d+)?)', response)
            if score_match:
                return float(score_match.group(1))
            
            # If not found, look for any number between -100 and 100
            number_match = re.search(r'-?\d+(?:\.\d+)?', response)
            if number_match:
                score = float(number_match.group())
                if -100 <= score <= 100:
                    return score
            
            print(f"Could not extract valid score from: {response}")
            return None
        except Exception as e:
            print(f"Error extracting score: {e}")
            return None
        

    def should_terminate(self) -> bool:
        # Implement early stopping logic here
        if len(self.output_data["rollouts"]) > 2:
            last_two_rewards = [r["reward"] for r in self.output_data["rollouts"][-2:]]
            if abs(last_two_rewards[0] - last_two_rewards[1]) < 1:
                return True
        return False

    def save_output(self):
        with open('mctsr_output.json', 'w') as f:
            json.dump(self.output_data, f, indent=2)

import sys
import argparse 


def main():
    # Default values
    default_model = "qwen2:0.5b-instruct-fp16"
    default_rollouts = 8
    default_problem = "If a rectangle has a length of 10 units and a width of 5 units, what is its area?"

    if len(sys.argv) > 1:  # If command-line arguments are provided
        parser = argparse.ArgumentParser(description="Run MCTSr for problem-solving")
        parser.add_argument("--model", type=str, default=default_model, help="Model to use for LLM API")
        parser.add_argument("--rollouts", type=int, default=default_rollouts, help="Number of rollouts to perform")
        parser.add_argument("--problem_file", type=str, help="Path to the file containing the problem")
        
        args = parser.parse_args()

        model = args.model
        rollouts = args.rollouts

        if args.problem_file:
            with open(args.problem_file, 'r') as file:
                problem = file.read().strip()
        else:
            problem = default_problem
    else:  # If no command-line arguments are provided
        model = default_model
        rollouts = default_rollouts
        problem = default_problem

    print(f"Problem: {problem}")
    print(f"Model: {model}")
    print(f"Number of rollouts: {rollouts}")
    print()

    llm_api = LLMApi()
    mctsr = MCTSr(llm_api, problem, max_rollouts=rollouts)
    
    best_answer = mctsr.run()
    print(f"\nBest answer: {best_answer}")

if __name__ == "__main__":
    main()