package com.capstone.backend.domain.intervention.service;

import com.capstone.backend.common.exception.EntityNotFoundException;
import com.capstone.backend.domain.emotion.entity.EmotionLog;
import com.capstone.backend.domain.emotion.repository.EmotionLogRepository;
import com.capstone.backend.domain.intervention.dto.FeedbackRequestDto;
import com.capstone.backend.domain.intervention.dto.InterventionRequestDto;
import com.capstone.backend.domain.intervention.dto.InterventionResponseDto;
import com.capstone.backend.domain.intervention.entity.Intervention;
import com.capstone.backend.domain.intervention.repository.InterventionRepository;
import com.capstone.backend.domain.user.entity.User;
import com.capstone.backend.domain.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class InterventionService {

    private final InterventionRepository interventionRepository;
    private final UserRepository userRepository;
    private final EmotionLogRepository emotionLogRepository;

    @Transactional
    public InterventionResponseDto saveIntervention(InterventionRequestDto dto) {
        User user = userRepository.findById(dto.userId())
                .orElseThrow(() -> new EntityNotFoundException("사용자를 찾을 수 없습니다. userId=" + dto.userId()));
        EmotionLog emotionLog = emotionLogRepository.findById(dto.emotionLogId())
                .orElseThrow(() -> new EntityNotFoundException("감정 로그를 찾을 수 없습니다. id=" + dto.emotionLogId()));

        int cooldownMin = dto.cooldownMinutes() != null ? dto.cooldownMinutes() : 30;
        LocalDateTime cooldownExpiresAt = LocalDateTime.now().plusMinutes(cooldownMin);

        Intervention intervention = Intervention.builder()
                .user(user)
                .emotionLog(emotionLog)
                .interventionType(dto.interventionType())
                .agentMessage(dto.agentMessage())
                .ttsPlayed(dto.ttsPlayed())
                .cooldownExpiresAt(cooldownExpiresAt)
                .build();
        return InterventionResponseDto.from(interventionRepository.save(intervention));
    }

    @Transactional(readOnly = true)
    public List<InterventionResponseDto> getInterventionsByUserId(Long userId) {
        return interventionRepository.findByUserUserIdOrderByCreatedAtDesc(userId)
                .stream()
                .map(InterventionResponseDto::from)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public boolean isCooldownActive(Long userId) {
        return interventionRepository.existsByUserUserIdAndCooldownExpiresAtAfter(userId, LocalDateTime.now());
    }

    @Transactional
    public InterventionResponseDto submitFeedback(Long interventionId, FeedbackRequestDto dto) {
        Intervention intervention = interventionRepository.findById(interventionId)
                .orElseThrow(() -> new EntityNotFoundException("개입 이력을 찾을 수 없습니다. id=" + interventionId));
        intervention.submitFeedback(dto.interventionRating(), dto.emotionAccuracy(), dto.feedbackText());
        return InterventionResponseDto.from(intervention);
    }
}
