package com.capstone.backend.domain.recommendation.repository;

import com.capstone.backend.domain.recommendation.entity.Recommendation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RecommendationRepository extends JpaRepository<Recommendation, Long> {
    List<Recommendation> findByInterventionInterventionId(Long interventionId);
}
