"""
python ganymede.py --model "qwen2:0.5b-instruct-fp16" --rollouts 10 --problem_file problem.txt

"""

import json
import math
import random
from typing import List, Dict, Any
from openai import OpenAI
import re
import logging
import sys
import argparse
import os

# Set up logging to write to a file
log_file_path = 'mctsr.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.error(f"Error in LLM API call: {e}")
            return ""

class MCTSrNode:
    def __init__(self, problem: str, answer: str, parent=None, is_dummy: bool = False):
        self.problem = problem
        self.answer = answer
        self.parent = parent
        self.children: List[MCTSrNode] = []
        self.visits = 0
        self.q_value = 0
        self.rewards: List[float] = []
        self.is_dummy = is_dummy

class MCTSr:
    def __init__(self, llm_api: LLMApi, problem: str, max_rollouts: int):
        self.llm_api = llm_api
        self.problem = problem
        self.max_rollouts = max_rollouts
        self.dummy_answers = [
            "I don't know.",
            "I can't understand this question.",
            "I can't help with this question.",
            "I don't know how to solve this question.",
            "I don't know the answer to this question.",
            "I don't know the answer to this question, sorry."
        ]
        self.root = self.initialize_root()
        self.output_data: Dict[str, Any] = {"problem": problem, "rollouts": []}
        self.c = 1  # UCB exploration parameter
        self.epsilon = 1e-6  # Small constant to avoid division by zero
        self.quality_threshold = 50  # Minimum quality threshold for termination
        self.dummy_penalty = 50  # Penalty for dummy answers

    def initialize_root(self) -> MCTSrNode:
        naive_answer = self.llm_api.generate(f"Solve this problem: {self.problem}")
        dummy_answer = random.choice(self.dummy_answers)
        
        root = MCTSrNode(self.problem, "")
        root.children = [
            MCTSrNode(self.problem, naive_answer, root),
            MCTSrNode(self.problem, dummy_answer, root, is_dummy=True)
        ]
        return root

    def run(self) -> str:
        for rollout in range(self.max_rollouts):
            self.print_formatted(f"Rollout {rollout + 1}/{self.max_rollouts}")
            node = self.select()
            new_node = self.expand(node)
            reward = self.evaluate(new_node)
            if reward is None:
                self.print_formatted(f"Failed to extract score in rollout {rollout + 1}. Stopping early.")
                break
            self.backpropagate(new_node, reward)

            self.output_data["rollouts"].append({
                "rollout_number": rollout + 1,
                "selected_answer": node.answer,
                "refined_answer": new_node.answer,
                "reward": reward
            })

            self.print_graph_state()

            if self.should_terminate():
                break

        best_answer = self.best_answer()
        self.output_data["best_answer"] = best_answer
        self.save_output()
        return best_answer

    def print_formatted(self, message: str):
        print(f"\n[MCTSr] {message}")

    def print_graph_state(self):
        print("\nCurrent Graph State:")
        self.print_node(self.root, "", True)

    def print_node(self, node: MCTSrNode, prefix: str, is_last: bool):
        node_info = f"Q: {node.q_value:.2f}, Visits: {node.visits}"
        if node.is_dummy:
            node_info += " (Dummy)"
        print(f"{prefix}{'└── ' if is_last else '├── '}{node_info}")
        
        prefix += "    " if is_last else "│   "
        for i, child in enumerate(node.children):
            self.print_node(child, prefix, i == len(node.children) - 1)

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
        dummy_penalty = self.dummy_penalty if node.is_dummy else 0
        return exploitation + self.c * exploration - dummy_penalty

    def expand(self, node: MCTSrNode) -> MCTSrNode:
        feedback = self.get_feedback(node)
        refined_answer = self.refine_answer(node, feedback)
        new_node = MCTSrNode(node.problem, refined_answer, parent=node)
        node.children.append(new_node)
        return new_node

    def evaluate(self, node: MCTSrNode) -> float:
        reward = self.get_reward(node)
        if reward is None:
            self.print_formatted(f"Failed to get valid reward for node: {node.answer}")
            return 0.0
        if node.is_dummy:
            reward = min(reward, -self.dummy_penalty)  # Ensure dummy answers always get a negative reward
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
        non_dummy_nodes = [n for n in self.root.children if not n.is_dummy]
        if non_dummy_nodes:
            return max(non_dummy_nodes, key=lambda n: n.q_value).answer
        self.print_formatted("All answers are dummy answers. Generating a new answer.")
        return self.llm_api.generate(f"Please solve this problem: {self.problem}")

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
        
        Output a score between -10 and +10 (e.g., from -10 to +10).
        
        Your response MUST end with the following format:
        [Score]: X
        Where X is the numerical score between -10 and +10.
        """
        max_retries = 3
        for _ in range(max_retries):
            response = self.llm_api.generate(prompt)
            score = self.extract_score(response)
            if score is not None:
                return min(score, 9)  # Implement full score suppression
        self.print_formatted(f"Failed to extract score after {max_retries} attempts. Last response: {response}")
        return None

    @staticmethod
    def extract_score(response: str) -> float:
        logging.info(f"Attempting to extract score from response")
        
        try:
            # Priority pattern for values in square brackets
            bracket_pattern = r'\[(-?\d+(?:\.\d+)?)\]'
            bracket_match = re.search(bracket_pattern, response)
            if bracket_match:
                score = float(bracket_match.group(1))
                logging.info(f"Extracted score from square brackets: {score}")
                return MCTSr.normalize_score(score)

            # Other score patterns as fallback
            score_patterns = [
                r'\[Score\]:\s*(-?\d+(?:\.\d+)?)',
                r'score\s+is\s+(-?\d+(?:\.\d+)?)',
                r'score:\s*(-?\d+(?:\.\d+)?)',
                r'(-?\d+(?:\.\d+)?)\s*/\s*10',
                r'give\s+a\s+score\s+of\s+(-?\d+(?:\.\d+)?)',
                r'final\s+score:\s*(-?\d+(?:\.\d+)?)',
                r'overall\s+score:\s*(-?\d+(?:\.\d+)?)'
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    logging.info(f"Extracted score using pattern '{pattern}': {score}")
                    return MCTSr.normalize_score(score)
            
            logging.warning(f"Could not find any valid score in the response")
            return None
        except Exception as e:
            logging.error(f"Error extracting score: {e}")
            return None

    @staticmethod
    def normalize_score(score: float) -> float:
        normalized = max(-10, min(10, score))
        if normalized != score:
            logging.info(f"Normalized score from {score} to {normalized}")
        return normalized

    def should_terminate(self) -> bool:
        best_node = max(self.root.children, key=lambda n: n.q_value)
        if best_node.is_dummy or best_node.q_value < self.quality_threshold:
            return False
        if len(self.output_data["rollouts"]) > 2:
            last_two_rewards = [r["reward"] for r in self.output_data["rollouts"][-2:]]
            if abs(last_two_rewards[0] - last_two_rewards[1]) < 1:
                return True
        return False

    def save_output(self):
        with open('mctsr_output.json', 'w') as f:
            json.dump(self.output_data, f, indent=2)

def main():
    # Default values
    default_model = "qwen2:0.5b-instruct-fp16"
    default_rollouts = 8
    default_problem = "If a rectangle has a length of 30 units and a width of 5 units, what is its area?"

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
    print(f"Logging detailed information to: {os.path.abspath(log_file_path)}")
    print()

    llm_api = LLMApi()
    mctsr = MCTSr(llm_api, problem, max_rollouts=rollouts)
    
    best_answer = mctsr.run()
    print(f"\nBest answer: {best_answer}")

if __name__ == "__main__":
    main()
