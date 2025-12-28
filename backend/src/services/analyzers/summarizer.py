from typing import List
from src.schemas.audio_metrics import (
    AudioFeatures,
    FeedbackSummary,
    SummarizedInterval,
    WarningEvent,
)

class FeedbackSummarizer:
    def summarize(self, metrics_list: List[AudioFeatures]) -> FeedbackSummary:
        if not metrics_list:
            return FeedbackSummary(
                total_duration=0.0,
                chunk_count=0,
                overall_score=0.0,
            )

        total_duration = metrics_list[-1].end_time - metrics_list[0].start_time
        chunk_count = len(metrics_list)

        success_chunks = []
        info_chunks = []
        warnings = []
        all_scores = []
        all_suggestions = set()

        for metrics in metrics_list:
            if not metrics.feedback:
                continue

            all_scores.append(metrics.feedback.overall_score)
            all_suggestions.update(metrics.feedback.suggestions)

            self._process_feedback_item(
                metrics, "pitch_dynamics", metrics.feedback.pitch_dynamics,
                success_chunks, info_chunks, warnings
            )
            self._process_feedback_item(
                metrics, "pitch_range", metrics.feedback.pitch_range,
                success_chunks, info_chunks, warnings
            )
            self._process_feedback_item(
                metrics, "fluency", metrics.feedback.fluency,
                success_chunks, info_chunks, warnings
            )
            self._process_feedback_item(
                metrics, "hesitation", metrics.feedback.hesitation,
                success_chunks, info_chunks, warnings
            )
            self._process_volume_item(
                metrics, metrics.feedback.volume,
                success_chunks, info_chunks, warnings
            )

        success_intervals = self._merge_intervals(success_chunks)
        info_intervals = self._merge_intervals(info_chunks)

        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return FeedbackSummary(
            total_duration=total_duration,
            chunk_count=chunk_count,
            overall_score=overall_score,
            success_intervals=success_intervals,
            info_intervals=info_intervals,
            warnings=warnings,
            suggestions=list(all_suggestions),
        )

    def _process_feedback_item(
        self, metrics, category, item,
        success_chunks, info_chunks, warnings
    ):
        chunk_data = {
            "start_time": metrics.start_time,
            "end_time": metrics.end_time,
            "category": category,
            "status": item.status,
            "message": item.message,
            "pitch_mean": metrics.pitch_mean,
            "pitch_std": metrics.pitch_std,
            "voiced_prob": metrics.voiced_prob,
            "volume": metrics.volume,
        }

        if item.severity == "success":
            success_chunks.append(chunk_data)
        elif item.severity == "info":
            info_chunks.append(chunk_data)
        elif item.severity == "warning":
            warnings.append(WarningEvent(
                timestamp=metrics.start_time,
                category=category,
                status=item.status,
                message=item.message,
            ))

    def _process_volume_item(
        self, metrics, item,
        success_chunks, info_chunks, warnings
    ):
        chunk_data = {
            "start_time": metrics.start_time,
            "end_time": metrics.end_time,
            "category": "volume",
            "status": item.status,
            "message": item.message,
            "pitch_mean": metrics.pitch_mean,
            "pitch_std": metrics.pitch_std,
            "voiced_prob": metrics.voiced_prob,
            "volume": metrics.volume,
        }

        if item.severity == "success":
            success_chunks.append(chunk_data)
        elif item.severity == "info":
            info_chunks.append(chunk_data)
        elif item.severity == "warning":
            warnings.append(WarningEvent(
                timestamp=metrics.start_time,
                category="volume",
                status=item.status,
                message=item.message,
                db_fs=item.db_fs,
            ))

    def _merge_intervals(self, chunks: List[dict]) -> List[SummarizedInterval]:
        if not chunks:
            return []

        grouped = {}
        for chunk in chunks:
            key = (chunk["category"], chunk["status"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(chunk)

        intervals = []
        for (category, status), group in grouped.items():
            merged = self._merge_consecutive(group)
            for m in merged:
                intervals.append(SummarizedInterval(
                    start_time=m["start_time"],
                    end_time=m["end_time"],
                    status=f"{category}: {status}",
                    message=m["message"],
                    avg_pitch_mean=m["avg_pitch_mean"],
                    avg_pitch_std=m["avg_pitch_std"],
                    avg_voiced_prob=m["avg_voiced_prob"],
                    avg_volume=m["avg_volume"],
                    chunk_count=m["chunk_count"],
                ))

        intervals.sort(key=lambda x: x.start_time)
        return intervals

    def _merge_consecutive(self, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []

        sorted_chunks = sorted(chunks, key=lambda x: x["start_time"])
        merged = []
        
        current = self._init_accumulator(sorted_chunks[0])

        for chunk in sorted_chunks[1:]:
            if chunk["start_time"] <= current["end_time"] + 0.1:
                current["end_time"] = max(current["end_time"], chunk["end_time"])
                current["pitch_means"].append(chunk["pitch_mean"])
                current["pitch_stds"].append(chunk["pitch_std"])
                current["voiced_probs"].append(chunk["voiced_prob"])
                current["volumes"].append(chunk["volume"])
                current["chunk_count"] += 1
            else:
                merged.append(self._finalize_interval(current))
                current = self._init_accumulator(chunk)

        merged.append(self._finalize_interval(current))
        return merged

    def _init_accumulator(self, chunk):
        return {
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "message": chunk["message"],
            "pitch_means": [chunk["pitch_mean"]],
            "pitch_stds": [chunk["pitch_std"]],
            "voiced_probs": [chunk["voiced_prob"]],
            "volumes": [chunk["volume"]],
            "chunk_count": 1,
        }

    def _finalize_interval(self, current: dict) -> dict:
        return {
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "message": current["message"],
            "avg_pitch_mean": sum(current["pitch_means"]) / len(current["pitch_means"]),
            "avg_pitch_std": sum(current["pitch_stds"]) / len(current["pitch_stds"]),
            "avg_voiced_prob": sum(current["voiced_probs"]) / len(current["voiced_probs"]),
            "avg_volume": sum(current["volumes"]) / len(current["volumes"]),
            "chunk_count": current["chunk_count"],
        }