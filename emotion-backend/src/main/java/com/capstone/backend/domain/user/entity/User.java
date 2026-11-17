package com.capstone.backend.domain.user.entity;

import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@Table(name = "users")
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@EntityListeners(AuditingEntityListener.class)
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_id")
    private Long userId;

    @Column(name = "login_id", unique = true, length = 50)
    private String loginId;

    @Column(length = 255)
    private String password;

    @Column(unique = true, length = 100)
    private String email;

    @Column(nullable = false, length = 50)
    private String nickname;

    private String address;

    @Column(name = "address_detail")
    private String addressDetail;

    private String zipcode;

    @CreatedDate
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @LastModifiedDate
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    public void update(String nickname, String address, String addressDetail, String zipcode) {
        this.nickname = nickname;
        this.address = address;
        this.addressDetail = addressDetail;
        this.zipcode = zipcode;
    }
}
