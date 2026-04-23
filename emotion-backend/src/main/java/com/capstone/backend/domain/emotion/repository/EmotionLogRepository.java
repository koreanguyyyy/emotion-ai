package com.capstone.backend.domain.emotion.repository;

import com.capstone.backend.domain.emotion.entity.EmotionLog;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface EmotionLogRepository extends JpaRepository<EmotionLog, Long> {
    List<EmotionLog> findByUserUserIdOrderByDetectedAtDesc(Long userId);
    List<EmotionLog> findByUserUserIdOrderByDetectedAtDesc(Long userId, Pageable pageable);
}
