package com.capstone.backend.domain.user.repository;

import com.capstone.backend.domain.user.entity.UserPreference;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserPreferenceRepository extends JpaRepository<UserPreference, Long> {
    Optional<UserPreference> findByUserUserId(Long userId);
}
