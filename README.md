# Rural Supply Chain Optimization System

A multi-agent AI system that optimizes rural supply-chain distribution in India. The system forecasts village-level demand, optimizes routing, monitors disruptions, and coordinates field operations across low-bandwidth networks.

## Architecture

The system follows an event-driven microservices pattern with six specialized agents:

- **Data Ingestion Agent**: Pulls and normalizes external data sources
- **Demand Forecasting Agent**: Trains village-level demand prediction models
- **Routing Optimization Agent**: Solves stock placement and vehicle routing problems
- **Disruption Monitoring Agent**: Monitors weather alerts and triggers replanning
- **Field Coordination Agent**: Manages field communications via SMS/IVR
- **Dashboard Agent**: Produces dashboards and reports for supply officers

## Technology Stack

- **Language**: Python 3.11+
- **Message Bus**: RabbitMQ
- **Databases**: PostgreSQL with PostGIS, TimescaleDB, Redis
- **Testing**: pytest, Hypothesis (property-based testing)
- **Optimization**: PuLP, Google OR-Tools
- **ML**: XGBoost, LightGBM
- **API**: FastAPI

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+

### Local Development

1. Start infrastructure services:
```bash
docker-compose up -d
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
pytest
```

4. Run property-based tests:
```bash
pytest -m property
```

## Project Structure

```
src/
├── agents/              # Agent implementations
│   ├── data_ingestion/
│   ├── demand_forecasting/
│   ├── routing_optimization/
│   ├── disruption_monitoring/
│   ├── field_coordination/
│   └── dashboard/
├── common/              # Shared utilities
│   ├── events.py       # Event definitions
│   ├── models.py       # Data models
│   └── message_bus.py  # Message bus client
└── config/              # Configuration

tests/
├── unit/               # Unit tests
├── property/           # Property-based tests
└── integration/        # Integration tests
```

## Development Phases

- **Phase 1 (MVP)**: Core forecasting → routing → replanning pipeline for single district
- **Phase 2**: Event-driven replanning, automated retraining, scenario generation
- **Phase 3**: SMS/IVR integration and offline field coordination
- **Phase 4**: Public sector data integration and equity constraints

## License

MIT
