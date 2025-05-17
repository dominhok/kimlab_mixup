import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from tokenizers import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

from config import ExperimentConfig
from prompts.templates import TEMPLATES, SYSTEM
from utils.metrics import evaluate_correction


SOLAR_PRO_TOKENIZER = Tokenizer.from_pretrained(
    "upstage/solar-pro-tokenizer"
)

TRAIN_DF = pd.read_csv("./data/train.csv")

TOKENIZED_ERR_SENTENCES = [SOLAR_PRO_TOKENIZER.encode(sent, add_special_tokens=False).ids 
                           for sent in TRAIN_DF["err_sentence"].array]


def construct_fewshot(dataset, tokenized_sentences, text, topk=10, chat_fewshot=True):
    tokenized_text = SOLAR_PRO_TOKENIZER.encode(text, add_special_tokens=False).ids
    sentences = [tokenized_text] + tokenized_sentences

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    topk_idxs = np.argsort(similarity[0])[::-1][:topk]
    fewshot = [(dataset.iloc[idx]["err_sentence"], dataset.iloc[idx]["cor_sentence"]) for idx in topk_idxs]
    if chat_fewshot:
        chat_fewshot = []
        for user_sentence, assistant_sentence in fewshot:
            chat_fewshot.append({"role": "user", "content": user_sentence})
            chat_fewshot.append({"role": "assistant", "content": assistant_sentence})
        fewshot = chat_fewshot
    return fewshot


def fewshot_prompt(fewshot, sentence):
    prompt = "다음은 입력과 출력 예시들입니다:\n\n"
    for err, cor in fewshot:
        prompt += f"입력: {err}\n"
        prompt += f"출력: {cor}\n\n"
    prompt += f"입력: {sentence}\n"
    prompt += "출력: "
    return prompt


def random_fewshot(dataset, topk=10, seed=None, chat_fewshot=True):
    if seed is not None:
        np.random.seed(seed)
    rand_idxs = np.random.choice(len(dataset), size=topk)
    fewshot = [(dataset.iloc[idx]["err_sentence"], dataset.iloc[idx]["cor_sentence"]) for idx in rand_idxs]
    if chat_fewshot:
        chat_fewshot = []
        for user_sentence, assistant_sentence in fewshot:
            chat_fewshot.append({"role": "user", "content": user_sentence})
            chat_fewshot.append({"role": "assistant", "content": assistant_sentence})
        fewshot = chat_fewshot
    return fewshot


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[config.template_name]
        self.api_url = config.api_url
        self.model = config.model
    
    def _make_prompt(self, text: str) -> str:
        """프롬프트 생성"""
        return self.template.format(text=text)
    
    def _call_api_single(self, prompt: str, seed=None) -> str:
        """단일 문장에 대한 API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "서사의 흐름이 역순행적으로 진행된다고 보는게 맞나요?"},
                {"role": "assistant", "content": "서사의 흐름이 역 순행적으로 진행된다고 보는 것이 맞나요?"},
                {"role": "user", "content": "근대 이런 시험을 마주할 때 중요한 건 시험 때도 중요하지만 이후 정신력 관리가 더 중요한것같아요."},
                {"role": "assistant", "content": "근데 이런 시험을 마주할 때 중요한 건 시험 때도 중요하지만 이후 정신력 관리가 더 중요한 것 같아요."},
                {"role": "user", "content": "속두와 속력에서공통점이잇다고..."},
                {"role": "assistant", "content": "속도와 속력에서 공통점이 있다고..."},
                {"role": "user", "content": "제가 자퇴 전 전교권에 들어본적이 있는데요"},
                {"role": "assistant", "content": "제가 자퇴 전에 전교권에 들어본 적이 있는데요."},
                {"role": "user", "content": "나가알아보면대는데이미 예약하셨다니 뭐..."},
                {"role": "assistant", "content": "내가 알아보면 되는데 이미 예약하셨다니 뭐..."},
                {"role": "user", "content": "아님 문제읽고 찾아서 푸는건가요"},
                {"role": "assistant", "content": "아니면 문제 읽고 찾아서 푸는 건가요?"},
                {"role": "user", "content": "없으면 예를 쫌 들어주새요."},
                {"role": "assistant", "content": "없으면 예를 좀 들어주세요."},
                {"role": "user", "content": "페이지스는거 깜빡해써요."},
                {"role": "assistant", "content": "페이지 쓰는 걸 깜빡했어요."},
                {"role": "user", "content": "혹시 재가 간과한게잇나?"},
                {"role": "assistant", "content": "혹시 제가 간과한 게 있나?"},
                {"role": "user", "content": "그러면 진공이아닌상태에서는 어떡게돼나요?"},
                {"role": "assistant", "content": "그러면 진공이 아닌 상태에서는 어떻게 되나요?"},
                *construct_fewshot(TRAIN_DF, TOKENIZED_ERR_SENTENCES, prompt, topk=10, chat_fewshot=True),
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        output = results["choices"][0]["message"]["content"]
        return output

    def run(self, data: pd.DataFrame, seed=None) -> pd.DataFrame:
        """데이터셋에 대한 실험 실행"""
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            prompt = self._make_prompt(row['err_sentence'])
            corrected = self._call_api_single(prompt, seed=seed)
            results.append({
                'id': row['id'],
                'cor_sentence': corrected
            })
        df = pd.DataFrame(results)
        return df

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None, seed=None) -> Dict:
        """템플릿별 실험 실행"""
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")
        
        # 학습 데이터로 실험
        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data, seed=seed)
        train_recall = evaluate_correction(train_data, train_results)
        
        # 검증 데이터로 실험
        print("\n[검증 데이터 실험]")
        valid_results = self.run(valid_data, seed=seed)
        valid_recall = evaluate_correction(valid_data, valid_results)
        
        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        } 