if ENABLE_XGBOOST and XGBRegressor is not None:
            tr_x = train if len(train) <= MAX_TRAIN_ROWS else train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
            best_x = None
            for params in XGB_GRID:
                mx = make_xgb(params, XGB_MAX_TREES, early_stopping=True)
                mx.fit(tr_x[inputs], tr_x[f"{t}_future"], eval_set=[(val[inputs], y_val)], verbose=False)
                n_best = int(getattr(mx, "best_iteration", XGB_MAX_TREES - 1)) + 1
                r = rmse(y_val, mx.predict(val[inputs], iteration_range=(0, n_best)))
                xgb_grid_rows.append({"Group": label, "Target": t, "Settings": str(params),
                                      "Trees_chosen": n_best, "RMSE_val": round(r, 3)})
                if best_x is None or r < best_x[0]:
                    best_x = (r, params, n_best)
            m, pt, secs = fit_predict(make_xgb(best_x[1], best_x[2]), inputs, trval, test, t)
            rows.append({"Run": RUN_LABEL, "Group": label, "Target": t, "Model": "XGBoost",
                         "RMSE_val": round(best_x[0], 3), **score(y_test, pt),
                         "Settings": str({**best_x[1], "n_estimators": best_x[2]}), "Train_sec": round(secs, 1)})
            preds[(t, "XGBoost")] = pt
            models_final[(t, "XGBoost")] = make_xgb(best_x[1], best_x[2])
            importances[(t, "XGB")] = pd.Series(m.feature_importances_, index=inputs)
            print(f"  {t}: best XGBoost settings on validation = {best_x[1]} with {best_x[2]} trees "
                  f"(validation RMSE {best_x[0]:.3f})")
    if ENABLE_LSTM:
        rows.extend(run_lstm(label, train, val_full, test_full, inputs))
    res = pd.DataFrame(rows)
    res["Test_rows"] = res["Test_rows"].fillna(len(test)).astype(int) if "Test_rows" in res.columns else len(test)
    pd.DataFrame(grid_rows).to_csv(os.path.join(BASE_DIR, f"out_rf_validation_{safe}_{RUN_LABEL}.csv"), index=False)
    if xgb_grid_rows:
        pd.DataFrame(xgb_grid_rows).to_csv(os.path.join(BASE_DIR, f"out_xgb_validation_{safe}_{RUN_LABEL}.csv"), index=False)
