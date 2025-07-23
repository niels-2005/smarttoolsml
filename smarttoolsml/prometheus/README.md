# Prometheus Monitoring Setup

This directory contains a complete Prometheus monitoring stack with Grafana for visualization and Alertmanager for notifications.

## Overview

This setup provides a comprehensive monitoring solution that includes:
- **Prometheus** server for metrics collection
- **Grafana** for visualization and dashboards
- **Alertmanager** for handling alerts and notifications
- **Exporters** for collecting system and application metrics

## Files Structure

### `docker-compose.yml`
Contains the Docker Compose configuration to run the complete monitoring stack with:
- Prometheus server
- Grafana dashboard
- Various exporters for metrics collection

### `prometheus.yml`
Main configuration file for Prometheus that includes:
- Scrape configurations for different exporters
- Target definitions for monitoring endpoints
- Global settings and intervals

### `alertmanager.yml`
Configuration file for Prometheus Alertmanager that defines:
- Alert rules and conditions
- Notification channels (email, Slack, etc.)
- Routing rules for different alert types

### `rules/` Directory
Contains alert rules files that will be loaded by Prometheus to:
- Evaluate metrics against defined thresholds
- Trigger alerts when conditions are met
- Define custom alerting logic

## Getting Started

1. **Start the monitoring stack:**
   ```bash
   docker-compose up -d
   ```

2. **Access the services:**
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`
   - Alertmanager: `http://localhost:9093`

3. **Configuration:**
   - Ensure all configuration files are properly mounted in your Docker setup
   - Customize the scrape targets in `prometheus.yml` according to your infrastructure
   - Configure alert rules in the `rules/` directory as needed

## Usage

Use this setup to monitor your applications and infrastructure metrics effectively. The combination of Prometheus, Grafana, and Alertmanager provides a complete observability solution for your systems.