
## 🧪 Model Training

The training pipeline (`train_models.py`) includes:

1. **Data Loading**: Merge exercise, calories, and weather data
2. **Feature Engineering**: Gender encoding, temperature adjustments
3. **Preprocessing**: StandardScaler normalization
4. **Model Training**: 7 algorithms with hyperparameter tuning
5. **Evaluation**: MAE, RMSE, R² metrics on test set
6. **Model Persistence**: Save to PKL files

## 📈 Performance Insights

- **Best Accuracy**: XGBoost & LightGBM (R² = 0.996)
- **Fastest Training**: Linear Regression (0.05s)
- **Best Balance**: LightGBM (high accuracy + fast training)
- **Key Features**: Duration and Heart Rate show strongest correlation with calories (r > 0.85)

## 🎓 Use Cases

- **Fitness Apps**: Integrate calorie prediction API
- **Wearable Devices**: Real-time calorie tracking
- **Health Research**: Analyze exercise efficiency
- **Personal Training**: Customize workout plans
- **ML Education**: Learn ensemble methods and explainability

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Dataset inspired by exercise science research
- SHAP library for model interpretability
- Streamlit for rapid prototyping
- scikit-learn for ML infrastructure


---

⭐ **Star this repo** if you found it helpful!

**Live Demo**: [click here for live demo]([https://your-app.streamlit.app](https://calorie-burn-full-ml-lifecycle-with-interactive-dashboard-nrnw.streamlit.app/))
