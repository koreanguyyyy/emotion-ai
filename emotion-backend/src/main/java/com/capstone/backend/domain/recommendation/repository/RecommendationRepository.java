package com.capstone.backend.domain.recommendation.repository;

import com.capstone.backend.domain.recommendation.entity.Recommendation;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface RecommendationRepository extends JpaRepository<Recommendation, Long> {
    List<Recommendation> findByInterventionInterventionId(Long interventionId);

    @Query("SELECT r FROM Recommendation r WHERE r.intervention.user.userId = :userId ORDER BY r.createdAt DESC")
    List<Recommendation> findRecentByUserId(@Param("userId") Long userId, Pageable pageable);
}
