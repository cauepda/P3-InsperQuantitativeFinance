import numpy as np
import pandas as pd

class auxiliar_functions():
    def __init__(self, start_date="2015-06-01", random_state=42, size=0):
        self.start_date = start_date
        self.random_state = random_state
        self.size = size if size > 0 else None
        np.random.seed(self.random_state)

    def load_total_ret_df(self, timef="daily"):
        df_total_ret = pd.read_csv('data/df_total_ret.csv')
        df_total_ret['Dates'] = pd.to_datetime(df_total_ret['Dates'])
        df_total_ret = df_total_ret.loc[df_total_ret["Dates"] >= self.start_date]
        df_total_ret = df_total_ret.loc[:, ~df_total_ret.columns.str.contains('Equity.1')]
        columns_to_keep = ['Dates']

        for col in df_total_ret.columns:
            if col != 'Dates':
                col_data = df_total_ret[col]
                null_count = col_data.isnull().sum()
                if null_count == 0:
                    columns_to_keep.append(col)

        df_total_ret = df_total_ret[columns_to_keep]

        if timef == "monthly":
            df_total_ret = self.resample_to_monthly(df_total_ret, date_column='Dates')

        return df_total_ret

    def get_random_stocks(self, df_total_ret):
        stocks = np.random.choice(df_total_ret.columns[1:], size=self.size, replace=False) if self.size else df_total_ret.columns[1:].tolist()
        return stocks

    def load_common_factors_df(self, include_momentum=True):
        """
        Load Fama-French 5-factor data and optionally add momentum factor
        """
        # Carregar fatores Fama-French 5 originais
        df_factors = pd.read_csv('data/F-F_Research_Data_5_Factors_2x3.csv', skiprows=3)
        df_factors = df_factors[:-1]
        df_factors = df_factors.dropna(subset=['Unnamed: 0'])
        df_factors = df_factors[df_factors['Unnamed: 0'].astype(str).str.len() == 6]
        df_factors = df_factors[df_factors['Unnamed: 0'].astype(str).str.isdigit()]
        df_factors['Date'] = pd.to_datetime(df_factors['Unnamed: 0'].astype(str), format='%Y%m')
        df_factors['Date_end_month'] = df_factors['Date'] + pd.offsets.MonthEnd(0)
        
        # Converter fatores para decimal
        factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for col in factor_columns:
            if col in df_factors.columns:
                df_factors[col] = pd.to_numeric(df_factors[col], errors='coerce') / 100
        
        df_factors = df_factors.drop('Unnamed: 0', axis=1)
        df_factors = df_factors.drop('Date', axis=1)
        df_factors = df_factors.rename(columns={'Date_end_month': 'Dates'})
        
        # Adicionar momentum se solicitado
        if include_momentum:
            momentum_factor = self._load_momentum_factor()
            if momentum_factor is not None:
                # Fazer merge com os fatores originais
                df_factors = pd.merge(df_factors, momentum_factor, on='Dates', how='left')
        
        # Filtrar por data e configurar índice
        df_factors = df_factors.loc[df_factors["Dates"] >= self.start_date]
        df_factors = df_factors.set_index('Dates')
        
        return df_factors
    
    def _load_momentum_factor(self):
        """
        Carrega e processa o fator momentum diário, convertendo para mensal
        """
        try:
            # Carregar dados diários de momentum
            df_momentum = pd.read_csv('data/F-F_Momentum_Factor_daily.csv', skiprows=12)
            
            # Limpar dados
            df_momentum = df_momentum.dropna()
            
            # Converter primeira coluna (datas) para datetime
            # Formato aparece como YYYYMMDD
            df_momentum.iloc[:, 0] = df_momentum.iloc[:, 0].astype(str)
            
            # Filtrar apenas datas válidas (8 dígitos)
            df_momentum = df_momentum[df_momentum.iloc[:, 0].str.len() == 8]
            df_momentum = df_momentum[df_momentum.iloc[:, 0].str.isdigit()]
            
            # Converter para datetime
            df_momentum['Date'] = pd.to_datetime(df_momentum.iloc[:, 0], format='%Y%m%d', errors='coerce')
            
            # Converter momentum para decimal (assumindo que vem em percentual)
            df_momentum['Mom'] = pd.to_numeric(df_momentum.iloc[:, 1], errors='coerce') / 100
            
            # Remover linhas com NaN
            df_momentum = df_momentum.dropna(subset=['Date', 'Mom'])
            
            # Converter para final de mês
            df_momentum['Date_end_month'] = df_momentum['Date'] + pd.offsets.MonthEnd(0)
            monthly_momentum = df_momentum.groupby('Date_end_month')['Mom'].last().reset_index()
            monthly_momentum = monthly_momentum.rename(columns={'Date_end_month': 'Dates', 'Mom': 'MOM'})
            
            print(f"✅ Momentum factor carregado: {len(monthly_momentum)} observações mensais")
            return monthly_momentum
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar momentum factor: {e}")
            print("Continuando apenas com os 5 fatores Fama-French originais")
            return None
    
    def load_mcap(self, timef="daily"):
        """
        Load market capitalization data
        """
        df_mkt_cap = pd.read_csv('data/df_mkt_cap.csv')
        df_mkt_cap['Dates'] = pd.to_datetime(df_mkt_cap['Dates'])
        df_mkt_cap = df_mkt_cap.loc[df_mkt_cap["Dates"] >= self.start_date]
        
        if timef == "monthly":
            df_mkt_cap = self.resample_to_monthly(df_mkt_cap, date_column='Dates')
        
        return df_mkt_cap
    
    def load_prices(self, timef="daily"):
        """
        Load price data
        """
        df_prices = pd.read_csv('data/df_prices.csv')
        df_prices['Dates'] = pd.to_datetime(df_prices['Dates'])
        df_prices = df_prices.loc[df_prices["Dates"] >= self.start_date]
        
        if timef == "monthly":
            df_prices = self.resample_to_monthly(df_prices, date_column='Dates')
        
        return df_prices
    
    def load_spx_membership(self):
        """
        Load S&P 500 membership data
        """
        df_spx_memb = pd.read_csv('data/df_spx_memb.csv')
        return df_spx_memb
    
    def resample_to_monthly(self, df, date_column='Dates'):
        """
        Resample daily data to monthly (last day of month)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with daily data
        date_column : str
            Name of the date column
            
        Returns:
        --------
        pd.DataFrame
            Monthly resampled data
        """
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_indexed = df_copy.set_index(date_column)
        monthly_data = df_indexed.resample('M').last()
        monthly_data = monthly_data.reset_index()
        return monthly_data
    
    def filter_by_date_range(self, df, start_date=None, end_date=None, date_column='Dates'):
        """
        Filter DataFrame by date range
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to filter
        start_date : str, optional
            Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date (format: 'YYYY-MM-DD')
        date_column : str
            Name of the date column
            
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        if start_date is not None:
            df_copy = df_copy[df_copy[date_column] >= start_date]
        if end_date is not None:
            df_copy = df_copy[df_copy[date_column] <= end_date]
            
        return df_copy
    
    def calculate_portfolio_returns(self, df_values, date_column='Date', value_column='Total_Value'):
        """
        Calculate portfolio returns from value series
        
        Parameters:
        -----------
        df_values : pd.DataFrame
            DataFrame with portfolio values
        date_column : str
            Name of the date column
        value_column : str
            Name of the value column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with returns added
        """
        df_copy = df_values.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.sort_values(date_column)
        df_copy['Returns'] = df_copy[value_column].pct_change()
        return df_copy
    
    def calculate_cumulative_returns(self, returns_series):
        """
        Calculate cumulative returns from return series
        
        Parameters:
        -----------
        returns_series : pd.Series
            Series of returns
            
        Returns:
        --------
        pd.Series
            Cumulative returns
        """
        return (1 + returns_series).cumprod() - 1
    
    def annualize_metrics(self, monthly_returns, rf_annual=0.025):
        r = monthly_returns.squeeze().dropna()
        rf_m = (1 + rf_annual) ** (1/12) - 1

        n = len(r)

        # CAGR anualizado (geométrico)
        growth = (1 + r).prod()
        annualized_return = growth ** (12 / n) - 1

        # Vol anualizada
        annualized_vol = r.std(ddof=1) * (12 ** 0.5)

        # Sharpe (forma simples e comum)
        sharpe = (annualized_return - rf_annual) / annualized_vol if annualized_vol != 0 else 0.0

        return {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "cagr_total_period": growth - 1,
            "n_months": n
        }
    
    def calculate_turnover(self, prev_shares, new_shares, current_prices, total_value):
        """
        Calculate portfolio turnover (one-way)
        
        Parameters:
        -----------
        prev_shares : np.array
            Previous shares held
        new_shares : np.array
            New shares to hold
        current_prices : np.array
            Current prices
        total_value : float
            Total portfolio value
            
        Returns:
        --------
        float
            One-way turnover
        """
        dollar_changes = np.abs(new_shares * current_prices - prev_shares * current_prices)
        turnover = dollar_changes.sum() / (2 * total_value)
        return turnover
    
    def calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        """
        Calculate Information Ratio vs benchmark
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns:
        --------
        dict
            Dictionary with IR and tracking error
        """
        portfolio_returns = portfolio_returns.dropna()
        benchmark_returns = benchmark_returns.dropna()
        
        # Align indices
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]
        
        # Calculate excess returns
        excess_returns = port_ret - bench_ret
        
        # Tracking error (annualized)
        tracking_error = excess_returns.std() * np.sqrt(12)
        
        # Information ratio (annualized)
        if tracking_error > 0:
            information_ratio = (excess_returns.mean() * 12) / tracking_error
        else:
            information_ratio = 0
            
        return {
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def calculate_comprehensive_stats(self, portfolio_returns, benchmark_returns=None, portfolio_name='Portfolio', rf_annual=0.025):
        """
        Calculate comprehensive portfolio statistics
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Monthly portfolio returns
        benchmark_returns : pd.Series, optional
            Monthly benchmark returns
        portfolio_name : str
            Name of the portfolio
        rf_annual : float
            Annual risk-free rate
            
        Returns:
        --------
        dict
            Dictionary with all statistics
        """
        # Basic annualized metrics
        metrics = self.annualize_metrics(portfolio_returns, rf_annual)
        
        stats = {
            'Portfolio': portfolio_name,
            'Annualized Average Return (%)': metrics['annualized_return'] * 100,
            'Annualized Average Std Dev (%)': metrics['annualized_volatility'] * 100,
            'Sharpe Ratio': metrics['sharpe_ratio']
        }
        
        # Add Information Ratio if benchmark provided
        if benchmark_returns is not None:
            ir_metrics = self.calculate_information_ratio(portfolio_returns, benchmark_returns)
            stats['Information Ratio vs S&P500'] = ir_metrics['information_ratio']
            stats['Tracking Error (%)'] = ir_metrics['tracking_error'] * 100
        else:
            stats['Information Ratio vs S&P500'] = 0
            
        return stats
    
    def backtest_equal_weight(self, monthly_prices, initial_balance=1_000_000.0, rf_monthly=None, benchmark_returns=None):
        """
        Equal-weighted, monthly-rebalanced backtest using pre-rebalance valuation
        
        Parameters:
        -----------
        monthly_prices : pd.DataFrame
            DataFrame with monthly prices (must have datetime index)
        initial_balance : float
            Initial portfolio value
        rf_monthly : pd.Series, optional
            Monthly risk-free rate series
        benchmark_returns : pd.Series, optional
            Monthly benchmark returns
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'returns': monthly portfolio returns
            - 'cum_return': cumulative return series
            - 'turnover': monthly turnover series
            - 'shares': shares held per month
            - 'values': portfolio value per month
            - 'weights': equal weights per month
            - 'metrics': annualized statistics
        """
        prices = monthly_prices.copy()
        prices = prices.sort_index()
        dates = prices.index
        tickers = prices.columns.to_list()
        n_assets = len(tickers)

        shares_records = []
        value_records = []
        weight_records = []
        turnover_records = []

        # Initial allocation
        p0 = prices.iloc[0].values
        dollar_per = initial_balance / n_assets
        shares = dollar_per / p0

        shares_records.append(pd.Series(shares, index=tickers, name=dates[0]))
        value0 = (shares * p0).sum()
        value_records.append((dates[0], value0))
        weight_records.append(pd.Series(np.ones(n_assets) / n_assets, index=tickers, name=dates[0]))
        turnover_records.append((dates[0], 1.0))

        # Monthly rebalancing loop
        for t in range(1, len(dates)):
            d = dates[t]
            p = prices.iloc[t].values

            current_values = shares * p
            total_value_before = current_values.sum()

            target_dollars = np.full(n_assets, total_value_before / n_assets)
            target_shares = target_dollars / p

            # Calculate turnover
            turnover = self.calculate_turnover(shares, target_shares, p, total_value_before)

            value_records.append((d, total_value_before))
            turnover_records.append((d, turnover))
            weight_records.append(pd.Series(np.ones(n_assets) / n_assets, index=tickers, name=d))

            shares = target_shares
            shares_records.append(pd.Series(shares, index=tickers, name=d))

        # Create output DataFrames
        df_shares = pd.DataFrame(shares_records)
        s_values = pd.Series({d: v for d, v in value_records}).sort_index()
        df_weights = pd.DataFrame(weight_records)
        s_turnover = pd.Series({d: t for d, t in turnover_records}).sort_index()

        # Calculate returns
        port_rets = s_values.pct_change().dropna()
        cum_rets = self.calculate_cumulative_returns(port_rets)

        # Calculate metrics
        metrics = self.annualize_metrics(port_rets)

        if rf_monthly is not None:
            rf = rf_monthly.reindex(port_rets.index).fillna(method="ffill")
            excess = port_rets - rf
            sharpe_ann = (excess.mean() * 12) / (port_rets.std(ddof=1) * np.sqrt(12))
            metrics["sharpe_annualized"] = sharpe_ann

        if benchmark_returns is not None:
            ir_metrics = self.calculate_information_ratio(port_rets, benchmark_returns)
            metrics["information_ratio_annualized"] = ir_metrics['information_ratio']
            metrics["tracking_error_annualized"] = ir_metrics['tracking_error']

        return {
            "returns": port_rets,
            "cum_return": cum_rets,
            "turnover": s_turnover,
            "shares": df_shares,
            "values": s_values,
            "weights": df_weights,
            "metrics": metrics,
        }
    
    def estimate_factor_model_covariance(
        self,
        returns_data, 
        factors_data,
        factor_list=None,
        use_excess=True,
        resid_floor_pct=20,
        ridge_ratio=1e-4,
        ensure_psd=True,
        return_r_squared=True,
        fit_intercept=True
    ):
        """
        UNIFIED: Combina versão simples + robustificações
        
        Parameters:
        -----------
        returns_data : DataFrame (T x N)
        factors_data : DataFrame (T x K)
        use_excess : bool
            Se True, converte para excess returns
        resid_floor_pct : float
            Percentil mínimo para variância idiossincrática (0=desligado)
        ridge_ratio : float
            Ridge regularization (0=desligado)
        ensure_psd : bool
            Força matriz PSD via eigenvalue fix
        return_r_squared : bool
            Se True, retorna R² das regressões
        fit_intercept : bool
            Se True, inclui α na regressão (depois descarta)
        """
        # Date alignment
        if factor_list is None:
            factor_list = factors_data.columns.tolist()
        idx = returns_data.index.intersection(factors_data.index)
        R = returns_data.loc[idx]
        F = factors_data.loc[idx, factor_list]
        
        # Excess returns (opcional)
        if use_excess and "RF" in factors_data.columns:
            rf = factors_data.loc[idx, "RF"]
            R = R.sub(rf, axis=0)
        
        # Normalize factors if in % (heuristic: mean > 0.5)
        F_scaled = F.copy()
        for col in F_scaled.columns:
            if F_scaled[col].abs().mean() > 0.5:
                F_scaled[col] = F_scaled[col] / 100.0
        
        # Regression loop
        betas_dic, resid_var, r_squared = {}, {}, {}
        K = F_scaled.shape[1]
        
        for col in R.columns:
            df_i = pd.concat([R[col], F_scaled], axis=1).dropna()
            if len(df_i) < (K + 6):
                continue
            
            y = df_i[col].values
            
            # OLS: com ou sem intercepto
            if fit_intercept:
                X = np.column_stack([np.ones(len(df_i)), df_i[F_scaled.columns].values])
                b, *_ = np.linalg.lstsq(X, y, rcond=None)
                betas_dic[col] = b[1:]  # Descarta intercepto
            else:
                X = df_i[F_scaled.columns].values
                b = np.linalg.pinv(X.T @ X) @ X.T @ y
                betas_dic[col] = b
            
            # Residuals
            resid = y - X @ b
            resid_var[col] = resid.var(ddof=K+1)
            
            # R² (opcional)
            if return_r_squared:
                ss_tot = ((y - y.mean())**2).sum()
                ss_res = (resid**2).sum()
                r_squared[col] = 1 - (ss_res / ss_tot)
        
        # Betas matrix
        betas = pd.DataFrame(betas_dic, index=F_scaled.columns).T
        B = betas.values
        
        # Factor covariance
        Sigma_f = np.cov(F_scaled.values, rowvar=False, ddof=1)
        
        # Idiosyncratic variance (com floor opcional)
        resid_var_series = pd.Series(resid_var).reindex(betas.index)
        if resid_floor_pct > 0:
            floor_val = np.percentile(resid_var_series.dropna(), resid_floor_pct)
            resid_var_series = resid_var_series.clip(lower=floor_val)
        D = np.diag(resid_var_series.values)
        
        # Reconstruct Σ = BFB' + D
        Sigma = B @ Sigma_f @ B.T + D
        
        # Ridge regularization (opcional)
        if ridge_ratio > 0:
            ridge_val = ridge_ratio * (np.trace(Sigma) / Sigma.shape[0])
            Sigma += ridge_val * np.eye(Sigma.shape[0])
        
        # PSD enforcement (opcional)
        if ensure_psd:
            vals, vecs = np.linalg.eigh((Sigma + Sigma.T) / 2)
            vals = np.maximum(vals, 1e-8 * np.trace(Sigma) / Sigma.shape[0])
            Sigma = vecs @ np.diag(vals) @ vecs.T
        
        # Output
        Sigma_df = pd.DataFrame(Sigma, index=betas.index, columns=betas.index)
        
        if return_r_squared:
            return Sigma_df, betas, pd.Series(r_squared)
        else:
            return Sigma_df, betas, resid_var_series
