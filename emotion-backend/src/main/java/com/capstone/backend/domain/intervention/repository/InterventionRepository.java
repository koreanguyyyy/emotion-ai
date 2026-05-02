package com.capstone.backend.domain.intervention.repository;

import com.capstone.backend.domain.intervention.entity.Intervention;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDateTime;
import java.util.List;

public interface InterventionRepository extends JpaRepository<Intervention, Long> {
    List<Intervention> findByUserUserIdOrderByCreatedAtDesc(Long userId);
    boolean existsByUserUserIdAndCooldownExpiresAtAfter(Long userId, LocalDateTime now);
}
