package com.capstone.backend.domain.user.service;

import com.capstone.backend.common.exception.EntityNotFoundException;
import com.capstone.backend.domain.user.dto.UserRequestDto;
import com.capstone.backend.domain.user.dto.UserResponseDto;
import com.capstone.backend.domain.user.entity.User;
import com.capstone.backend.domain.user.entity.UserPreference;
import com.capstone.backend.domain.user.repository.UserPreferenceRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserPreferenceService {

    private final UserPreferenceRepository userPreferenceRepository;

    @Transactional
    public UserPreference createDefault(User user) {
        UserPreference preference = UserPreference.builder()
                .user(user)
                .build();
        return userPreferenceRepository.save(preference);
    }

    @Transactional
    public UserResponseDto.PreferenceInfo updatePreferences(Long userId, UserRequestDto.UpdatePreference dto) {
        UserPreference preference = userPreferenceRepository.findByUserUserId(userId)
                .orElseThrow(() -> new EntityNotFoundException("선호 설정을 찾을 수 없습니다. userId=" + userId));
        preference.update(
                dto.preferredMusicGenres(),
                dto.preferredMovieGenres(),
                dto.preferredBookGenres(),
                dto.preferredPlaceTypes(),
                dto.ttsEnabled(),
                dto.cooldownMinutes()
        );
        return UserResponseDto.PreferenceInfo.from(preference);
    }
}
