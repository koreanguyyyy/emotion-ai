package com.capstone.backend.domain.recommendation.service;

import com.capstone.backend.common.exception.EntityNotFoundException;
import com.capstone.backend.domain.intervention.entity.Intervention;
import com.capstone.backend.domain.intervention.repository.InterventionRepository;
import com.capstone.backend.domain.recommendation.dto.RecommendationRequestDto;
import com.capstone.backend.domain.recommendation.dto.RecommendationResponseDto;
import com.capstone.backend.domain.recommendation.entity.Recommendation;
import com.capstone.backend.domain.recommendation.repository.RecommendationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class RecommendationService {

    private final RecommendationRepository recommendationRepository;
    private final InterventionRepository interventionRepository;

    @Transactional
    public RecommendationResponseDto saveRecommendation(RecommendationRequestDto.Create dto) {
        Intervention intervention = interventionRepository.findById(dto.interventionId())
                .orElseThrow(() -> new EntityNotFoundException("개입 이력을 찾을 수 없습니다. id=" + dto.interventionId()));
        Recommendation recommendation = Recommendation.builder()
                .intervention(intervention)
                .category(dto.category())
                .title(dto.title())
                .artistOrAuthor(dto.artistOrAuthor())
                .externalId(dto.externalId())
                .externalUrl(dto.externalUrl())
                .thumbnailUrl(dto.thumbnailUrl())
                .reason(dto.reason())
                .build();
        return RecommendationResponseDto.from(recommendationRepository.save(recommendation));
    }

    @Transactional(readOnly = true)
    public List<RecommendationResponseDto> getRecommendationsByInterventionId(Long interventionId) {
        return recommendationRepository.findByInterventionInterventionId(interventionId)
                .stream()
                .map(RecommendationResponseDto::from)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<RecommendationResponseDto> getRecentRecommendationsByUserId(Long userId, int size) {
        return recommendationRepository.findRecentByUserId(userId, PageRequest.of(0, size))
                .stream()
                .map(RecommendationResponseDto::from)
                .collect(Collectors.toList());
    }

    @Transactional
    public RecommendationResponseDto submitFeedback(Long recommendationId, RecommendationRequestDto.Feedback dto) {
        Recommendation recommendation = recommendationRepository.findById(recommendationId)
                .orElseThrow(() -> new EntityNotFoundException("추천을 찾을 수 없습니다. id=" + recommendationId));
        recommendation.submitFeedback(dto.userFeedback(), dto.relevanceRating(), dto.played(), dto.feedbackText());
        return RecommendationResponseDto.from(recommendation);
    }
}
