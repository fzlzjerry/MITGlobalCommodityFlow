# MITGlobalCommodityFlow

MITGlobalCommodityFlow is a global forecasting model developed by MIT for predicting the flow of imported commodities across multiple major ports in the United States. This project leverages Recurrent Neural Networks (RNNs) to analyze and forecast commodity flows, aiming to enhance the resilience and efficiency of the nationwide supply chain network.

## Table of Contents

- [Background](#background)
- [Business Problem & Task](#business-problem--task)
- [Data Description](#data-description)
- [Contributing](#contributing)
- [License](#license)

## Background

To improve the resilience of the nationwide supply chain network, it is essential to understand the processes at the most upstream part of goods flow in the USA – the ocean ports. By estimating the volume, category, value, and routing of goods, stakeholders can better plan transportation, allocate resources, and manage the ordering and sourcing of goods.

## Business Problem & Task

The US port system is a complex network where different commodities are imported through various ports to minimize ground transportation. The project aims to determine if a global forecasting model can effectively predict the weight of imported commodities across ten major US ports.

## Data Description

- **Ports**: Nine major US ports, including Savannah, New York/New Jersey, Boston, Norfolk, Houston, Los Angeles/Long Beach, Oakland, and Seattle/Tacoma.
- **Commodities**: Sixteen food commodities categorized using the 2-digit HS code.
- **Import Data**: Monthly import data from 2003 to 2024, including customs data, containerized vessel value (in USD), and containerized vessel weight (in kg).

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of your changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Create a pull request detailing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
