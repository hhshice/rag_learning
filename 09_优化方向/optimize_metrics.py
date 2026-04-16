# -*- coding: utf-8 -*-
"""
RAG 评估指标计算示例

核心指标：
1. 召回率 (Recall): 召回了多少相关文档
2. 精度 (Precision): 召回文档中有多少相关
3. MRR: 第一个相关文档的位置倒数
4. NDCG: 考虑排序位置的指标

安装依赖：pip install numpy
"""

from typing import List, Dict
import numpy as np


class RAGEvaluator:
    """RAG 系统评估器"""
    
    @staticmethod
    def calculate_recall(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算召回率
        
        召回率 = 召回的相关文档 / 总相关文档
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表
            relevant_ids: 相关文档 ID 列表
        
        Returns:
            召回率 (0-1)
        
        示例:
            >>> evaluator.calculate_recall(['d1', 'd2', 'd3'], ['d1', 'd3', 'd5'])
            0.667  # 召回了 d1, d3 两个相关文档，总共 3 个相关文档
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # 召回的相关文档数量
        retrieved_relevant = len(retrieved_set & relevant_set)
        
        return retrieved_relevant / len(relevant_set)
    
    @staticmethod
    def calculate_precision(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算精度
        
        精度 = 召回的相关文档 / 总召回文档
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表
            relevant_ids: 相关文档 ID 列表
        
        Returns:
            精度 (0-1)
        
        示例:
            >>> evaluator.calculate_precision(['d1', 'd2', 'd3'], ['d1', 'd3', 'd5'])
            0.667  # 召回了 2 个相关文档（d1, d3），总共召回 3 个文档
        """
        if not retrieved_ids:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # 召回的相关文档数量
        retrieved_relevant = len(retrieved_set & relevant_set)
        
        return retrieved_relevant / len(retrieved_set)
    
    @staticmethod
    def calculate_f1(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算 F1 分数
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表
            relevant_ids: 相关文档 ID 列表
        
        Returns:
            F1 分数 (0-1)
        """
        precision = RAGEvaluator.calculate_precision(retrieved_ids, relevant_ids)
        recall = RAGEvaluator.calculate_recall(retrieved_ids, relevant_ids)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def calculate_mrr(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)
        
        MRR = 1 / 第一个相关文档的排名
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表
            relevant_ids: 相关文档 ID 列表
        
        Returns:
            MRR 分数 (0-1)
        
        示例:
            >>> evaluator.calculate_mrr(['d2', 'd1', 'd3'], ['d1', 'd5'])
            0.5  # d1 排在第 2 位，MRR = 1/2
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def calculate_ndcg(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """
        计算 NDCG (Normalized Discounted Cumulative Gain)
        
        NDCG = DCG / IDCG
        
        其中：
        DCG = Σ (2^rel_i - 1) / log2(i + 1)
        IDCG = 理想情况下的 DCG
        
        Args:
            retrieved_ids: 检索返回的文档 ID 列表
            relevant_ids: 相关文档 ID 列表
            k: 只考虑前 k 个结果
        
        Returns:
            NDCG 分数 (0-1)
        """
        relevant_set = set(relevant_ids)
        
        # 计算 DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                # 二值相关性：相关=1，不相关=0
                rel = 1.0
                dcg += rel / np.log2(i + 2)  # i+2 因为 log2(1) = 0
        
        # 计算 IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            rel = 1.0
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def calculate_map(
        query_results: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]]
    ) -> float:
        """
        计算 MAP (Mean Average Precision)
        
        MAP = 平均每个查询的 AP
        AP = Σ (Precision@k * rel_k) / 相关文档总数
        
        Args:
            query_results: {query: [retrieved_ids]}
            ground_truth: {query: [relevant_ids]}
        
        Returns:
            MAP 分数 (0-1)
        """
        aps = []
        
        for query, retrieved_ids in query_results.items():
            if query not in ground_truth:
                continue
            
            relevant_ids = ground_truth[query]
            relevant_set = set(relevant_ids)
            
            # 计算 AP (Average Precision)
            precisions = []
            num_relevant = 0
            
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_set:
                    num_relevant += 1
                    precision_at_k = num_relevant / (i + 1)
                    precisions.append(precision_at_k)
            
            if relevant_ids:
                ap = sum(precisions) / len(relevant_ids)
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def evaluate(
        self,
        query_results: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict:
        """
        完整评估
        
        Args:
            query_results: {query: [retrieved_ids]}
            ground_truth: {query: [relevant_ids]}
            k_values: 计算哪些 k 值的指标
        
        Returns:
            评估结果字典
        """
        metrics = {
            'recall': {},
            'precision': {},
            'f1': {},
            'mrr': [],
            'ndcg': {}
        }
        
        # 初始化
        for k in k_values:
            metrics['recall'][f'recall@{k}'] = []
            metrics['precision'][f'precision@{k}'] = []
            metrics['f1'][f'f1@{k}'] = []
            metrics['ndcg'][f'ndcg@{k}'] = []
        
        # 逐 Query 计算
        for query, retrieved_ids in query_results.items():
            if query not in ground_truth:
                continue
            
            relevant_ids = ground_truth[query]
            
            # 各 k 值的指标
            for k in k_values:
                metrics['recall'][f'recall@{k}'].append(
                    self.calculate_recall(retrieved_ids[:k], relevant_ids)
                )
                metrics['precision'][f'precision@{k}'].append(
                    self.calculate_precision(retrieved_ids[:k], relevant_ids)
                )
                metrics['f1'][f'f1@{k}'].append(
                    self.calculate_f1(retrieved_ids[:k], relevant_ids)
                )
                metrics['ndcg'][f'ndcg@{k}'].append(
                    self.calculate_ndcg(retrieved_ids, relevant_ids, k)
                )
            
            metrics['mrr'].append(
                self.calculate_mrr(retrieved_ids, relevant_ids)
            )
        
        # 计算平均值
        result = {
            'num_queries': len([q for q in query_results if q in ground_truth])
        }
        
        for k in k_values:
            result[f'recall@{k}'] = np.mean(metrics['recall'][f'recall@{k}'])
            result[f'precision@{k}'] = np.mean(metrics['precision'][f'precision@{k}'])
            result[f'f1@{k}'] = np.mean(metrics['f1'][f'f1@{k}'])
            result[f'ndcg@{k}'] = np.mean(metrics['ndcg'][f'ndcg@{k}'])
        
        result['mrr'] = np.mean(metrics['mrr'])
        result['map'] = self.calculate_map(query_results, ground_truth)
        
        return result
    
    def print_report(self, results: Dict):
        """打印评估报告"""
        print("=" * 70)
        print("📊 RAG 系统评估报告")
        print("=" * 70)
        
        print(f"\n查询数量: {results['num_queries']}\n")
        
        # 表头
        print(f"{'指标':<15} {'@5':<12} {'@10':<12} {'@20':<12}")
        print("-" * 70)
        
        # 召回率
        print(f"{'Recall':<15} "
              f"{results.get('recall@5', 0):.3f}{'':<7} "
              f"{results.get('recall@10', 0):.3f}{'':<7} "
              f"{results.get('recall@20', 0):.3f}")
        
        # 精度
        print(f"{'Precision':<15} "
              f"{results.get('precision@5', 0):.3f}{'':<7} "
              f"{results.get('precision@10', 0):.3f}{'':<7} "
              f"{results.get('precision@20', 0):.3f}")
        
        # F1
        print(f"{'F1':<15} "
              f"{results.get('f1@5', 0):.3f}{'':<7} "
              f"{results.get('f1@10', 0):.3f}{'':<7} "
              f"{results.get('f1@20', 0):.3f}")
        
        # NDCG
        print(f"{'NDCG':<15} "
              f"{results.get('ndcg@5', 0):.3f}{'':<7} "
              f"{results.get('ndcg@10', 0):.3f}{'':<7} "
              f"{results.get('ndcg@20', 0):.3f}")
        
        print("\n" + "-" * 70)
        print(f"MRR:  {results.get('mrr', 0):.3f}")
        print(f"MAP:  {results.get('map', 0):.3f}")
        
        print("\n" + "=" * 70)


def main():
    """演示评估流程"""
    print("\n" + "=" * 70)
    print("📈 RAG 评估指标演示")
    print("=" * 70)
    
    # 模拟数据
    query_results = {
        'query1': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7'],
        'query2': ['doc3', 'doc1', 'doc6', 'doc7', 'doc8', 'doc2', 'doc9'],
        'query3': ['doc2', 'doc4', 'doc9', 'doc10', 'doc1', 'doc3', 'doc5'],
        'query4': ['doc5', 'doc6', 'doc1', 'doc2', 'doc3', 'doc4', 'doc7'],
        'query5': ['doc1', 'doc3', 'doc5', 'doc7', 'doc9', 'doc2', 'doc4']
    }
    
    ground_truth = {
        'query1': ['doc1', 'doc3', 'doc5', 'doc7'],
        'query2': ['doc1', 'doc3', 'doc6'],
        'query3': ['doc2', 'doc4', 'doc9', 'doc10'],
        'query4': ['doc1', 'doc2', 'doc5', 'doc6'],
        'query5': ['doc1', 'doc3', 'doc5', 'doc7']
    }
    
    # 创建评估器
    evaluator = RAGEvaluator()
    
    # 单个指标示例
    print("\n【单个指标计算示例】")
    print("-" * 70)
    
    retrieved = query_results['query1'][:5]
    relevant = ground_truth['query1']
    
    recall = evaluator.calculate_recall(retrieved, relevant)
    precision = evaluator.calculate_precision(retrieved, relevant)
    f1 = evaluator.calculate_f1(retrieved, relevant)
    mrr = evaluator.calculate_mrr(retrieved, relevant)
    ndcg = evaluator.calculate_ndcg(retrieved, relevant, k=5)
    
    print(f"\n查询: query1")
    print(f"召回文档 (前5): {retrieved}")
    print(f"相关文档: {relevant}")
    print(f"\nRecall@5:    {recall:.3f}")
    print(f"Precision@5: {precision:.3f}")
    print(f"F1@5:        {f1:.3f}")
    print(f"MRR:         {mrr:.3f}")
    print(f"NDCG@5:      {ndcg:.3f}")
    
    # 完整评估
    print("\n【完整评估】")
    results = evaluator.evaluate(query_results, ground_truth, k_values=[5, 10, 20])
    evaluator.print_report(results)
    
    # 对比不同检索策略
    print("\n【检索策略对比】")
    print("=" * 70)
    
    # 策略1：仅向量检索（模拟）
    vector_results = {
        'query1': ['doc1', 'doc2', 'doc3', 'doc8', 'doc9'],
        'query2': ['doc3', 'doc7', 'doc8', 'doc9', 'doc10'],
        'query3': ['doc2', 'doc5', 'doc6', 'doc7', 'doc8']
    }
    
    # 策略2：混合检索（模拟，效果更好）
    hybrid_results = {
        'query1': ['doc1', 'doc3', 'doc5', 'doc2', 'doc7'],
        'query2': ['doc1', 'doc3', 'doc6', 'doc2', 'doc7'],
        'query3': ['doc2', 'doc4', 'doc9', 'doc10', 'doc1']
    }
    
    ground_truth_compare = {
        'query1': ['doc1', 'doc3', 'doc5', 'doc7'],
        'query2': ['doc1', 'doc3', 'doc6'],
        'query3': ['doc2', 'doc4', 'doc9', 'doc10']
    }
    
    print("\n策略1：仅向量检索")
    results1 = evaluator.evaluate(vector_results, ground_truth_compare, k_values=[5])
    
    print("\n策略2：混合检索")
    results2 = evaluator.evaluate(hybrid_results, ground_truth_compare, k_values=[5])
    
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)
    print(f"\n{'指标':<15} {'向量检索':<15} {'混合检索':<15} {'提升':<15}")
    print("-" * 70)
    
    for metric in ['recall@5', 'precision@5', 'f1@5', 'ndcg@5', 'mrr', 'map']:
        v1 = results1.get(metric, 0)
        v2 = results2.get(metric, 0)
        improve = (v2 - v1) / v1 * 100 if v1 > 0 else 0
        
        print(f"{metric:<15} {v1:.3f}{'':<10} {v2:.3f}{'':<10} {improve:+.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ 评估演示完成")
    print("=" * 70)
    
    print("\n💡 关键点:")
    print("1. 建立标准评估数据集（Query + 相关文档）")
    print("2. 多维度评估（Recall、Precision、MRR、NDCG）")
    print("3. A/B 对比不同策略")
    print("4. 持续监控和优化")


if __name__ == "__main__":
    main()
