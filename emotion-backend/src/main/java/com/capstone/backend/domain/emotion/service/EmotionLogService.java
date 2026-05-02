package com.capstone.backend.domain.emotion.service;

import com.capstone.backend.common.exception.EntityNotFoundException;
import com.capstone.backend.domain.emotion.dto.EmotionLogRequestDto;
import com.capstone.backend.domain.emotion.dto.EmotionLogResponseDto;
import com.capstone.backend.domain.emotion.entity.EmotionLog;
import com.capstone.backend.domain.emotion.repository.EmotionLogRepository;
import com.capstone.backend.domain.user.entity.User;
import com.capstone.backend.domain.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class EmotionLogService {

    private final EmotionLogRepository emotionLogRepository;
    private final UserRepository userRepository;

    @Transactional
    public EmotionLogResponseDto saveEmotionLog(EmotionLogRequestDto dto) {
        User user = userRepository.findById(dto.userId())
                .orElseThrow(() -> new EntityNotFoundException("사용자를 찾을 수 없습니다. userId=" + dto.userId()));
        EmotionLog log = EmotionLog.builder()
                .user(user)
                .faceEmotion(dto.faceEmotion())
                .faceConfidence(dto.faceConfidence())
                .speechText(dto.speechText())
                .vadStressScore(dto.vadStressScore())
                .finalStressLevel(dto.finalStressLevel())
                .sighDetected(dto.sighDetected() != null ? dto.sighDetected() : false)
                .motionPattern(dto.motionPattern())
                .interventionTriggered(dto.interventionTriggered())
                .detectedAt(LocalDateTime.now())
                .build();
        return EmotionLogResponseDto.from(emotionLogRepository.save(log));
    }

    @Transactional(readOnly = true)
    public List<EmotionLogResponseDto> getEmotionLogsByUserId(Long userId) {
        return emotionLogRepository.findByUserUserIdOrderByDetectedAtDesc(userId)
                .stream()
                .map(EmotionLogResponseDto::from)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<EmotionLogResponseDto> getRecentEmotionLogs(Long userId, int count) {
        return emotionLogRepository.findByUserUserIdOrderByDetectedAtDesc(userId, PageRequest.of(0, count))
                .stream()
                .map(EmotionLogResponseDto::from)
                .collect(Collectors.toList());
    }
}
