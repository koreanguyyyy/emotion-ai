package com.capstone.backend.domain.emotion.repository;

import com.capstone.backend.domain.emotion.entity.EmotionDailySummary;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.Optional;

public interface EmotionDailySummaryRepository extends JpaRepository<EmotionDailySummary, Long> {
    Optional<EmotionDailySummary> findByUserUserIdAndSummaryDate(Long userId, LocalDate summaryDate);
}
