-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Aug 01, 2026 at 06:55 PM
-- Server version: 9.1.0
-- PHP Version: 8.3.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `proforma.et.db`
--

-- --------------------------------------------------------

--
-- Table structure for table `login_attempts`
--

DROP TABLE IF EXISTS `login_attempts`;
CREATE TABLE IF NOT EXISTS `login_attempts` (
  `a_id` int NOT NULL,
  `time` int NOT NULL,
  `user_id` varchar(255) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `proforma_team`
--

DROP TABLE IF EXISTS `proforma_team`;
CREATE TABLE IF NOT EXISTS `proforma_team` (
  `pt_id` int NOT NULL AUTO_INCREMENT,
  `proforma` varchar(32) DEFAULT NULL,
  `member_id` int DEFAULT NULL,
  PRIMARY KEY (`pt_id`),
  KEY `proforma` (`proforma`),
  KEY `member_id` (`member_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `proforma_team`
--

INSERT INTO `proforma_team` (`pt_id`, `proforma`, `member_id`) VALUES
(1, 'ebb1853083590e0b08b3434afa2e4d27', 1);

-- --------------------------------------------------------

--
-- Table structure for table `tbl_category`
--

DROP TABLE IF EXISTS `tbl_category`;
CREATE TABLE IF NOT EXISTS `tbl_category` (
  `category_id` int NOT NULL AUTO_INCREMENT,
  `category_name` varchar(100) NOT NULL,
  PRIMARY KEY (`category_id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_category`
--

INSERT INTO `tbl_category` (`category_id`, `category_name`) VALUES
(1, 'Construction'),
(2, 'IT Services'),
(3, 'Consulting'),
(4, 'Supply'),
(5, 'Maintenance');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_proforma`
--

DROP TABLE IF EXISTS `tbl_proforma`;
CREATE TABLE IF NOT EXISTS `tbl_proforma` (
  `proforma_id` varchar(32) NOT NULL,
  `proforma_name` varchar(255) NOT NULL,
  `category` int DEFAULT NULL,
  `organization` varchar(32) DEFAULT NULL,
  `created_on` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `status` varchar(50) DEFAULT 'draft',
  PRIMARY KEY (`proforma_id`),
  KEY `category` (`category`),
  KEY `organization` (`organization`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_proforma`
--

INSERT INTO `tbl_proforma` (`proforma_id`, `proforma_name`, `category`, `organization`, `created_on`, `status`) VALUES
('ebb1853083590e0b08b3434afa2e4d27', 'Test', 2, '87487d04d26542510161fdd8eb11f3b4', '2026-08-01 18:52:28', 'draft');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_proforma_items`
--

DROP TABLE IF EXISTS `tbl_proforma_items`;
CREATE TABLE IF NOT EXISTS `tbl_proforma_items` (
  `item_id` int NOT NULL AUTO_INCREMENT,
  `item_name` varchar(255) NOT NULL,
  `quantity` int NOT NULL,
  `unit` varchar(50) NOT NULL,
  `description` text,
  `proforma` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`item_id`),
  KEY `proforma` (`proforma`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_proforma_items`
--

INSERT INTO `tbl_proforma_items` (`item_id`, `item_name`, `quantity`, `unit`, `description`, `proforma`) VALUES
(1, 'PC', 2, 'Unit', 'TEst', 'ebb1853083590e0b08b3434afa2e4d27');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_provider_admin`
--

DROP TABLE IF EXISTS `tbl_provider_admin`;
CREATE TABLE IF NOT EXISTS `tbl_provider_admin` (
  `pa_id` varchar(32) NOT NULL,
  `email` varchar(191) NOT NULL,
  `first_name` varchar(100) DEFAULT NULL,
  `last_name` varchar(100) DEFAULT NULL,
  `profile_picture` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`pa_id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_seeker_address`
--

DROP TABLE IF EXISTS `tbl_seeker_address`;
CREATE TABLE IF NOT EXISTS `tbl_seeker_address` (
  `address_id` int NOT NULL AUTO_INCREMENT,
  `seeker_id` varchar(32) DEFAULT NULL,
  `address_line1` varchar(255) DEFAULT NULL,
  `address_line2` varchar(255) DEFAULT NULL,
  `city` varchar(100) DEFAULT NULL,
  `state` varchar(100) DEFAULT NULL,
  `postal_code` varchar(20) DEFAULT NULL,
  `country` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`address_id`),
  KEY `seeker_id` (`seeker_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_seeker_address`
--

INSERT INTO `tbl_seeker_address` (`address_id`, `seeker_id`, `address_line1`, `address_line2`, `city`, `state`, `postal_code`, `country`) VALUES
(2, '87487d04d26542510161fdd8eb11f3b4', NULL, NULL, NULL, NULL, NULL, NULL);

-- --------------------------------------------------------

--
-- Table structure for table `tbl_seeker_manager`
--

DROP TABLE IF EXISTS `tbl_seeker_manager`;
CREATE TABLE IF NOT EXISTS `tbl_seeker_manager` (
  `manager_id` varchar(32) NOT NULL,
  `first_name` varchar(100) NOT NULL,
  `last_name` varchar(100) NOT NULL,
  `email` varchar(191) NOT NULL,
  `phone` varchar(20) NOT NULL,
  `seeker` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`manager_id`),
  UNIQUE KEY `email` (`email`),
  KEY `seeker` (`seeker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_seeker_manager`
--

INSERT INTO `tbl_seeker_manager` (`manager_id`, `first_name`, `last_name`, `email`, `phone`, `seeker`) VALUES
('ef952cf3f9bc9878e2fb33d7e9389f1d', 'Abebe', 'Kebede', 'bamlew@gmail.com', '123', '87487d04d26542510161fdd8eb11f3b4');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_service_seeker`
--

DROP TABLE IF EXISTS `tbl_service_seeker`;
CREATE TABLE IF NOT EXISTS `tbl_service_seeker` (
  `seeker_id` varchar(32) NOT NULL,
  `seeker_name` varchar(255) NOT NULL,
  PRIMARY KEY (`seeker_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_service_seeker`
--

INSERT INTO `tbl_service_seeker` (`seeker_id`, `seeker_name`) VALUES
('87487d04d26542510161fdd8eb11f3b4', 'ABC');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_team`
--

DROP TABLE IF EXISTS `tbl_team`;
CREATE TABLE IF NOT EXISTS `tbl_team` (
  `team_id` int NOT NULL AUTO_INCREMENT,
  `first_name` varchar(100) NOT NULL,
  `last_name` varchar(100) NOT NULL,
  `email` varchar(191) NOT NULL,
  `phone` varchar(20) NOT NULL,
  `position` varchar(100) NOT NULL,
  `organization` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`team_id`),
  KEY `organization` (`organization`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_team`
--

INSERT INTO `tbl_team` (`team_id`, `first_name`, `last_name`, `email`, `phone`, `position`, `organization`) VALUES
(1, 'A', 'E', 'e@g', '213', '123', '87487d04d26542510161fdd8eb11f3b4');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_user`
--

DROP TABLE IF EXISTS `tbl_user`;
CREATE TABLE IF NOT EXISTS `tbl_user` (
  `user_id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `email` varchar(191) NOT NULL,
  `password` varchar(255) NOT NULL,
  `role` enum('manager','p_admin') NOT NULL,
  `user_information_id` varchar(32) NOT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `tbl_user`
--

INSERT INTO `tbl_user` (`user_id`, `username`, `email`, `password`, `role`, `user_information_id`) VALUES
(2, 'Abebe', 'bamlew@gmail.com', '$2y$10$KKpwbhFkbHZte6hkaoo2k.9jnInEvJLFheSZv0m7.n7KmlnRW1Acq', 'manager', 'ef952cf3f9bc9878e2fb33d7e9389f1d');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_vendor`
--

DROP TABLE IF EXISTS `tbl_vendor`;
CREATE TABLE IF NOT EXISTS `tbl_vendor` (
  `vendor_id` int NOT NULL AUTO_INCREMENT,
  `manager` varchar(32) DEFAULT NULL,
  `organization_name` varchar(255) DEFAULT NULL,
  `tin_number` varchar(50) DEFAULT NULL,
  `phone_number` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`vendor_id`),
  KEY `manager` (`manager`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `proforma_team`
--
ALTER TABLE `proforma_team`
  ADD CONSTRAINT `proforma_team_ibfk_1` FOREIGN KEY (`proforma`) REFERENCES `tbl_proforma` (`proforma_id`),
  ADD CONSTRAINT `proforma_team_ibfk_2` FOREIGN KEY (`member_id`) REFERENCES `tbl_team` (`team_id`);

--
-- Constraints for table `tbl_proforma`
--
ALTER TABLE `tbl_proforma`
  ADD CONSTRAINT `tbl_proforma_ibfk_1` FOREIGN KEY (`category`) REFERENCES `tbl_category` (`category_id`),
  ADD CONSTRAINT `tbl_proforma_ibfk_2` FOREIGN KEY (`organization`) REFERENCES `tbl_service_seeker` (`seeker_id`);

--
-- Constraints for table `tbl_proforma_items`
--
ALTER TABLE `tbl_proforma_items`
  ADD CONSTRAINT `tbl_proforma_items_ibfk_1` FOREIGN KEY (`proforma`) REFERENCES `tbl_proforma` (`proforma_id`);

--
-- Constraints for table `tbl_seeker_address`
--
ALTER TABLE `tbl_seeker_address`
  ADD CONSTRAINT `tbl_seeker_address_ibfk_1` FOREIGN KEY (`seeker_id`) REFERENCES `tbl_service_seeker` (`seeker_id`);

--
-- Constraints for table `tbl_seeker_manager`
--
ALTER TABLE `tbl_seeker_manager`
  ADD CONSTRAINT `tbl_seeker_manager_ibfk_1` FOREIGN KEY (`seeker`) REFERENCES `tbl_service_seeker` (`seeker_id`);

--
-- Constraints for table `tbl_team`
--
ALTER TABLE `tbl_team`
  ADD CONSTRAINT `tbl_team_ibfk_1` FOREIGN KEY (`organization`) REFERENCES `tbl_service_seeker` (`seeker_id`);

--
-- Constraints for table `tbl_vendor`
--
ALTER TABLE `tbl_vendor`
  ADD CONSTRAINT `tbl_vendor_ibfk_1` FOREIGN KEY (`manager`) REFERENCES `tbl_seeker_manager` (`manager_id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
