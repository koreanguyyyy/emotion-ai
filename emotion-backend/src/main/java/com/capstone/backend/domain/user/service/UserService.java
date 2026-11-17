package com.capstone.backend.domain.user.service;

import com.capstone.backend.common.exception.EntityNotFoundException;
import com.capstone.backend.domain.user.dto.UserRequestDto;
import com.capstone.backend.domain.user.dto.UserResponseDto;
import com.capstone.backend.domain.user.entity.User;
import com.capstone.backend.domain.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final UserPreferenceService userPreferenceService;

    @Transactional
    public UserResponseDto.Info createUser(UserRequestDto.Create dto) {
        if (userRepository.existsByLoginId(dto.loginId())) {
            throw new IllegalArgumentException("이미 사용 중인 아이디입니다.");
        }
        if (userRepository.existsByEmail(dto.email())) {
            throw new IllegalArgumentException("이미 사용 중인 이메일입니다.");
        }
        User user = User.builder()
                .loginId(dto.loginId())
                .password(dto.password())
                .email(dto.email())
                .nickname(dto.nickname())
                .address(dto.address())
                .addressDetail(dto.addressDetail())
                .zipcode(dto.zipcode())
                .build();
        User saved = userRepository.save(user);
        userPreferenceService.createDefault(saved);
        return UserResponseDto.Info.from(saved);
    }

    @Transactional(readOnly = true)
    public UserResponseDto.Info getUserById(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new EntityNotFoundException("사용자를 찾을 수 없습니다. userId=" + userId));
        return UserResponseDto.Info.from(user);
    }
}
